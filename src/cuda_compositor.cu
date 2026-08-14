#include "cuda_compositor.h"

#if RTA_WITH_CUDA

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>

#include "tiny_font.h"
#include "transform_geom.h"

namespace rta {
namespace {

#define RTA_CUDA_OK(expr)                       \
  do {                                          \
    const cudaError_t _e = (expr);              \
    if (_e != cudaSuccess) return false;        \
  } while (0)

inline cudaStream_t asStream(void* s) { return static_cast<cudaStream_t>(s); }

// Same palette as the CPU diagnostics overlay.
__constant__ float kPalette[8][3] = {
    {1.0f, 0.20f, 0.25f}, {0.20f, 0.85f, 1.0f}, {1.0f, 0.85f, 0.15f},
    {0.35f, 1.0f, 0.35f}, {1.0f, 0.45f, 1.0f},  {0.55f, 0.55f, 1.0f},
    {1.0f, 0.60f, 0.20f}, {0.20f, 1.0f, 0.75f},
};

__global__ void clearKernel(float* dst, std::ptrdiff_t stride, int x0, int y0, int w, int h) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  float* p = dst + stride * (y0 + y) + std::ptrdiff_t(x0 + x) * 4;
  p[0] = p[1] = p[2] = p[3] = 0.0f;
}

// Device mirror of GroupTransform, POD so it uploads with one memcpy.
struct DevTap {
  float offsetX, offsetY, scale, rotation, blur, opacity;
  int visible;
};

__device__ inline void discOffsetDev(int tap, int taps, float* dx, float* dy) {
  const float golden = 2.39996322972865332f;
  const float a = float(tap) * golden;
  const float r = sqrtf((float(tap) + 0.5f) / float(max(1, taps)));
  *dx = r * __cosf(a);
  *dy = r * __sinf(a);
}

// Label-masked bilinear fetch. Masking before filtering is what keeps
// antialiased glyph edges intact and stops a moving word dragging in a
// fragment of its neighbour.
__device__ inline void sampleMaskedDev(const float* src, std::ptrdiff_t srcStride,
                                       const int32_t* labels, const int32_t* labelToGroup,
                                       int labWidth, int imgW, int imgH, int group, float fx,
                                       float fy, float* out) {
  out[0] = out[1] = out[2] = out[3] = 0.0f;
  const int x0 = int(floorf(fx));
  const int y0 = int(floorf(fy));
  const float tx = fx - float(x0);
  const float ty = fy - float(y0);
  const float wx[2] = {1.0f - tx, tx};
  const float wy[2] = {1.0f - ty, ty};

  for (int j = 0; j < 2; ++j) {
    const int sy = y0 + j;
    if (sy < 0 || sy >= imgH) continue;
    for (int i = 0; i < 2; ++i) {
      const int sx = x0 + i;
      if (sx < 0 || sx >= imgW) continue;
      const int32_t lab = labels[std::ptrdiff_t(sy) * labWidth + sx];
      if (lab == 0 || labelToGroup[lab] != group) continue;
      const float wgt = wx[i] * wy[j];
      if (wgt <= 0.0f) continue;
      const float* p = src + srcStride * sy + std::ptrdiff_t(sx) * 4;
      out[0] += p[0] * wgt;
      out[1] += p[1] * wgt;
      out[2] += p[2] * wgt;
      out[3] += p[3] * wgt;
    }
  }
}

// One thread per tile pixel; each thread averages every shutter tap.
//
// Writes the group into its OWN tile rather than straight over the frame,
// because the defocus that follows has to see the sprite alone -- blurring into
// the frame would drag whatever is already composited under it into the result.
__global__ void renderTileKernel(float* tile, const float* src, std::ptrdiff_t srcStride,
                                 const int32_t* labels, const int32_t* labelToGroup, int labWidth,
                                 int imgW, int imgH, int group, float cx, float cy,
                                 const DevTap* taps, int nTaps, int dx0, int dy0, int dw, int dh) {
  const int lx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ly = blockIdx.y * blockDim.y + threadIdx.y;
  if (lx >= dw || ly >= dh) return;

  const int x = dx0 + lx;
  const int y = dy0 + ly;

  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int k = 0; k < nTaps; ++k) {
    const DevTap t = taps[k];
    if (!t.visible || t.opacity <= 0.0f) continue;

    // Inverse of  p' = centre + R(theta)*scale*(p - centre) + offset
    const float inv = 1.0f / (fabsf(t.scale) < 1e-6f ? 1e-6f : t.scale);
    const float c = __cosf(-t.rotation) * inv;
    const float s = __sinf(-t.rotation) * inv;
    const float ddx = float(x) + 0.5f - cx - t.offsetX;
    const float ddy = float(y) + 0.5f - cy - t.offsetY;
    const float fx = c * ddx - s * ddy + cx - 0.5f;
    const float fy = s * ddx + c * ddy + cy - 0.5f;

    float smp[4];
    sampleMaskedDev(src, srcStride, labels, labelToGroup, labWidth, imgW, imgH, group, fx, fy,
                    smp);
    if (smp[3] <= 0.0f) continue;
    acc[0] += smp[0] * t.opacity;
    acc[1] += smp[1] * t.opacity;
    acc[2] += smp[2] * t.opacity;
    acc[3] += smp[3] * t.opacity;
  }

  const float invTaps = 1.0f / float(nTaps);
  float* o = tile + (std::ptrdiff_t(ly) * dw + lx) * 4;
  o[0] = acc[0] * invTaps;
  o[1] = acc[1] * invTaps;
  o[2] = acc[2] * invTaps;
  o[3] = acc[3] * invTaps;
}

// One thread per LINE, walking it with a running window. Mirrors the CPU
// version's arithmetic exactly -- same additions in the same order -- so the two
// paths cannot drift apart in the last bits.
__global__ void boxPassKernel(const float* in, float* out, int w, int h, int r, int horizontal) {
  const int n = horizontal ? w : h;
  const int lines = horizontal ? h : w;
  const int l = blockIdx.x * blockDim.x + threadIdx.x;
  if (l >= lines) return;

  const std::ptrdiff_t step = horizontal ? 4 : std::ptrdiff_t(w) * 4;
  const std::ptrdiff_t lineStep = horizontal ? std::ptrdiff_t(w) * 4 : 4;
  const float inv = 1.0f / float(2 * r + 1);
  const float* ip = in + lineStep * l;
  float* op = out + lineStep * l;

  float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int i = 0; i <= r && i < n; ++i)
    for (int c = 0; c < 4; ++c) sum[c] += ip[step * i + c];

  for (int i = 0; i < n; ++i) {
    for (int c = 0; c < 4; ++c) op[step * i + c] = sum[c] * inv;
    const int add = i + r + 1, drop = i - r;
    if (add < n)
      for (int c = 0; c < 4; ++c) sum[c] += ip[step * add + c];
    if (drop >= 0)
      for (int c = 0; c < 4; ++c) sum[c] -= ip[step * drop + c];
  }
}

// Premultiplied "over" of the finished tile onto the frame.
__global__ void blendTileKernel(float* dst, std::ptrdiff_t dstStride, const float* tile, int dx0,
                                int dy0, int dw, int dh) {
  const int lx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ly = blockIdx.y * blockDim.y + threadIdx.y;
  if (lx >= dw || ly >= dh) return;

  const float* in = tile + (std::ptrdiff_t(ly) * dw + lx) * 4;
  const float a = in[3];
  if (a <= 0.0f) return;
  float* o = dst + dstStride * (dy0 + ly) + std::ptrdiff_t(dx0 + lx) * 4;
  const float ia = 1.0f - a;
  o[0] = in[0] + o[0] * ia;
  o[1] = in[1] + o[1] * ia;
  o[2] = in[2] + o[2] * ia;
  o[3] = a + o[3] * ia;
}

__global__ void dimKernel(float* dst, std::ptrdiff_t dstStride, const float* src,
                          std::ptrdiff_t srcStride, int x0, int y0, int w, int h, float factor) {
  const int lx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ly = blockIdx.y * blockDim.y + threadIdx.y;
  if (lx >= w || ly >= h) return;
  const int x = x0 + lx, y = y0 + ly;
  const float* s = src + srcStride * y + std::ptrdiff_t(x) * 4;
  float* d = dst + dstStride * y + std::ptrdiff_t(x) * 4;
  for (int c = 0; c < 4; ++c) d[c] = fmaxf(d[c], s[c] * factor);
}

// One thread per pixel of the outline's bounding band; writes only on the edge.
//
// The sides lean with the type: at row y the group spans
// [sx1 + shear, sx2 + shear] with shear = (y - yMid) * slantTan. Mirrors the CPU
// drawDiagnostics exactly -- the two paths must not disagree about where a word
// was found.
__global__ void boxKernel(float* dst, std::ptrdiff_t dstStride, int imgW, int imgH, float sx1,
                          float sx2, float slantTan, int yMid, int by1, int by2, int thickness,
                          int colourIndex, int pad, int winX1, int winY1, int winX2, int winY2) {
  const int lx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ly = blockIdx.y * blockDim.y + threadIdx.y;

  const int oy1 = by1 - thickness;
  const int h = (by2 + thickness) - oy1;
  const int y = oy1 + ly;
  if (ly >= h) return;

  const float shear = float(y - yMid) * slantTan;
  const int left = int(sx1 + shear);
  const int right = int(sx2 + shear);

  const int ox1 = left - thickness - pad;
  const int w = (right + thickness + pad) - ox1;
  if (lx >= w) return;
  const int x = ox1 + lx;

  if (x < winX1 || y < winY1 || x >= winX2 || y >= winY2) return;
  if (x < 0 || y < 0 || x >= imgW || y >= imgH) return;

  const bool onSide = (x >= left - thickness && x < left) ||
                      (x >= right - 1 && x < right + thickness);
  const bool onCap = (y < by1 + thickness - 1) || (y >= by2 - thickness + 1);
  const bool inSpan = x >= left - thickness && x < right + thickness;
  if (!(onSide || (onCap && inSpan))) return;

  const float* c = kPalette[colourIndex & 7];
  float* p = dst + dstStride * y + std::ptrdiff_t(x) * 4;
  p[0] = c[0];
  p[1] = c[1];
  p[2] = c[2];
  p[3] = 1.0f;
}

// Stamps a group's index above its box, matching the CPU diagnostics. Both paths
// must number the groups identically -- the numbers are what a manual merge or
// split names, so a disagreement would silently retarget an override.
__global__ void indexKernel(float* dst, std::ptrdiff_t dstStride, int imgW, int imgH, int ox,
                            int oy, int value, int digits, int scale, int colourIndex, int winX1,
                            int winY1, int winX2, int winY2) {
  const int lx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ly = blockIdx.y * blockDim.y + threadIdx.y;

  const int gw = digits * (4 * scale) - scale, gh = 5 * scale;
  if (lx >= gw + 2 * scale || ly >= gh + 2 * scale) return;

  const int x = ox - scale + lx, y = oy - scale + ly;
  if (x < winX1 || y < winY1 || x >= winX2 || y >= winY2) return;
  if (x < 0 || y < 0 || x >= imgW || y >= imgH) return;

  // Dark plate first: a bare digit over white type is invisible.
  float r = 0.0f, g = 0.0f, b = 0.0f;

  const int cx = lx - scale, cy = ly - scale;
  if (cx >= 0 && cy >= 0 && cy < gh) {
    const int cell = cx / (4 * scale);
    const int inCell = cx - cell * (4 * scale);
    if (cell < digits && inCell < 3 * scale) {
      int v = value;
      for (int k = digits - 1; k > cell; --k) v /= 10;
      if (digitPixel(v % 10, inCell, cy, scale)) {
        r = kPalette[colourIndex & 7][0];
        g = kPalette[colourIndex & 7][1];
        b = kPalette[colourIndex & 7][2];
      }
    }
  }

  float* p = dst + dstStride * y + std::ptrdiff_t(x) * 4;
  p[0] = r;
  p[1] = g;
  p[2] = b;
  p[3] = 1.0f;
}

// Subsamples every 8th pixel in both axes, matching the CPU hash's sampling.
__global__ void hashKernel(const float* src, std::ptrdiff_t stride, int width, int height,
                           float threshold, unsigned long long* acc) {
  const int gx = blockIdx.x * blockDim.x + threadIdx.x;
  const int gy = blockIdx.y * blockDim.y + threadIdx.y;
  const int x = gx * 8, y = gy * 8;
  if (x >= width || y >= height) return;

  const float a = src[stride * y + std::ptrdiff_t(x) * 4 + 3];
  const unsigned int q = (unsigned int)(fminf(15.0f, fmaxf(0.0f, a) * 15.0f));
  unsigned long long h = (unsigned long long)(q + (a > threshold ? 16u : 0u));

  // Mix in the position so a moved word changes the hash even if the histogram
  // of alpha values does not.
  h ^= (unsigned long long)(x) * 0x9E3779B97F4A7C15ull;
  h ^= (unsigned long long)(y) * 0xC2B2AE3D27D4EB4Full;
  h *= 0xFF51AFD7ED558CCDull;
  h ^= h >> 33;

  atomicAdd(acc, h);
}

RectI intersect(const RectI& a, const RectI& b) {
  RectI r{std::max(a.x1, b.x1), std::max(a.y1, b.y1), std::min(a.x2, b.x2), std::min(a.y2, b.y2)};
  return r.empty() ? RectI{0, 0, 0, 0} : r;
}

dim3 gridFor(int w, int h, dim3 block) {
  return dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
}

}  // namespace

bool cudaAvailable() {
  static const bool ok = [] {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
  }();
  return ok;
}

CudaSegmentation::~CudaSegmentation() { reset(); }

void CudaSegmentation::reset() {
  if (labels_) cudaFree(labels_);
  if (labelToGroup_) cudaFree(labelToGroup_);
  labels_ = nullptr;
  labelToGroup_ = nullptr;
  width_ = height_ = labelCount_ = 0;
}

bool CudaSegmentation::upload(const Segmentation& seg) {
  if (seg.labelImage.empty() || seg.labelToGroup.empty()) return false;

  const size_t pixels = size_t(seg.width) * seg.height;
  if (labels_ == nullptr || seg.width != width_ || seg.height != height_) {
    if (labels_) cudaFree(labels_);
    labels_ = nullptr;
    RTA_CUDA_OK(cudaMalloc(&labels_, pixels * sizeof(int32_t)));
    width_ = seg.width;
    height_ = seg.height;
  }
  if (labelToGroup_ == nullptr || int(seg.labelToGroup.size()) != labelCount_) {
    if (labelToGroup_) cudaFree(labelToGroup_);
    labelToGroup_ = nullptr;
    RTA_CUDA_OK(cudaMalloc(&labelToGroup_, seg.labelToGroup.size() * sizeof(int32_t)));
    labelCount_ = int(seg.labelToGroup.size());
  }

  RTA_CUDA_OK(cudaMemcpy(labels_, seg.labelImage.data(), pixels * sizeof(int32_t),
                         cudaMemcpyHostToDevice));
  RTA_CUDA_OK(cudaMemcpy(labelToGroup_, seg.labelToGroup.data(),
                         seg.labelToGroup.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
  return true;
}

bool cudaDownload(std::vector<float>& out, const float* srcDev, std::ptrdiff_t srcStrideFloats,
                  int width, int height) {
  if (!srcDev || width <= 0 || height <= 0) return false;

  // cudaMemcpy2D cannot take a negative pitch, so copy against the natural
  // layout: walk from the row with the lowest address using |stride|.
  const std::ptrdiff_t absStride = srcStrideFloats < 0 ? -srcStrideFloats : srcStrideFloats;
  const float* base = srcStrideFloats < 0 ? srcDev + srcStrideFloats * (height - 1) : srcDev;

  out.resize(size_t(absStride) * height);
  RTA_CUDA_OK(cudaMemcpy2D(out.data(), size_t(absStride) * sizeof(float), base,
                           size_t(absStride) * sizeof(float), size_t(width) * 4 * sizeof(float),
                           size_t(height), cudaMemcpyDeviceToHost));
  return true;
}

CudaScratch::~CudaScratch() {
  if (accum_) cudaFree(accum_);
  if (taps_) cudaFree(taps_);
  for (int i = 0; i < 2; ++i)
    if (tiles_[i]) cudaFree(tiles_[i]);
  accum_ = nullptr;
  taps_ = nullptr;
  tiles_[0] = tiles_[1] = nullptr;
}

void* CudaScratch::tileBuffer(int which, size_t bytes) {
  if (bytes == 0 || which < 0 || which > 1) return nullptr;
  if (tiles_[which] && tileBytes_[which] >= bytes) return tiles_[which];
  if (tiles_[which]) cudaFree(tiles_[which]);
  tiles_[which] = nullptr;
  tileBytes_[which] = 0;
  // Round up, so walking through groups of slightly different sizes does not
  // reallocate on every one of them.
  const size_t want = bytes * 2;
  if (cudaMalloc(&tiles_[which], want) != cudaSuccess) {
    tiles_[which] = nullptr;
    return nullptr;
  }
  tileBytes_[which] = want;
  return tiles_[which];
}

void* CudaScratch::tapBuffer(size_t bytes) {
  if (bytes == 0) return nullptr;
  if (taps_ && tapBytes_ >= bytes) return taps_;
  if (taps_) cudaFree(taps_);
  taps_ = nullptr;
  tapBytes_ = 0;
  // Round up so a changing group count doesn't reallocate every frame.
  const size_t want = bytes * 2;
  if (cudaMalloc(&taps_, want) != cudaSuccess) {
    taps_ = nullptr;
    return nullptr;
  }
  tapBytes_ = want;
  return taps_;
}

unsigned long long* CudaScratch::hashAccum() {
  if (!tried_) {
    tried_ = true;
    if (cudaMalloc(&accum_, sizeof(unsigned long long)) != cudaSuccess) accum_ = nullptr;
  }
  return accum_;
}

uint64_t cudaHashAlpha(const float* srcDev, std::ptrdiff_t srcStride, int width, int height,
                       float threshold, CudaScratch& scratch, void* stream, bool* ok) {
  if (ok) *ok = false;
  if (!srcDev || width <= 0 || height <= 0) return 0;

  unsigned long long* acc = scratch.hashAccum();
  if (!acc) return 0;

  cudaStream_t s = asStream(stream);
  uint64_t host = 0;
  bool good = cudaMemsetAsync(acc, 0, sizeof(unsigned long long), s) == cudaSuccess;
  if (good) {
    const dim3 block(16, 16);
    const int sw = (width + 7) / 8, sh = (height + 7) / 8;
    hashKernel<<<gridFor(sw, sh, block), block, 0, s>>>(srcDev, srcStride, width, height,
                                                        threshold, acc);
    good = cudaGetLastError() == cudaSuccess &&
           cudaMemcpyAsync(&host, acc, sizeof(host), cudaMemcpyDeviceToHost, s) == cudaSuccess &&
           cudaStreamSynchronize(s) == cudaSuccess;
  }

  // Fold in the dimensions so a resize always invalidates.
  host ^= (uint64_t(width) << 32) ^ uint64_t(height);
  if (ok) *ok = good;
  return host;
}

bool cudaClearWindow(float* dstDev, std::ptrdiff_t dstStride, int width, int height,
                     const RectI& window, void* stream) {
  if (!dstDev) return false;
  const RectI win = intersect(window, RectI{0, 0, width, height});
  if (win.empty()) return true;
  const dim3 block(16, 16);
  clearKernel<<<gridFor(win.width(), win.height(), block), block, 0, asStream(stream)>>>(
      dstDev, dstStride, win.x1, win.y1, win.width(), win.height());
  RTA_CUDA_OK(cudaGetLastError());
  return true;
}

bool cudaCompositeGroups(float* dstDev, std::ptrdiff_t dstStride, const float* srcDev,
                         std::ptrdiff_t srcStride, int width, int height,
                         const CudaSegmentation& devSeg, const std::vector<Group>& groups,
                         const std::vector<GroupTransform>& transforms, int taps,
                         const RectI& window, CudaScratch& scratch, void* stream,
                         const float* pivots) {
  taps = std::max(1, taps);
  if (!dstDev || !devSeg.valid() || transforms.size() != groups.size() * size_t(taps))
    return false;

  const RectI win = intersect(window, RectI{0, 0, width, height});
  if (win.empty()) return true;

  cudaStream_t s = asStream(stream);
  const dim3 block(16, 16);

  clearKernel<<<gridFor(win.width(), win.height(), block), block, 0, s>>>(
      dstDev, dstStride, win.x1, win.y1, win.width(), win.height());

  // Upload every tap in one transfer rather than per group.
  std::vector<DevTap> host(transforms.size());
  for (size_t i = 0; i < transforms.size(); ++i) {
    const GroupTransform& t = transforms[i];
    host[i] = DevTap{t.offsetX, t.offsetY, t.scale, t.rotation, t.blur, t.opacity, t.visible ? 1 : 0};
  }
  const size_t bytes = host.size() * sizeof(DevTap);
  DevTap* devTaps = static_cast<DevTap*>(scratch.tapBuffer(bytes));
  if (!devTaps) return false;
  RTA_CUDA_OK(cudaMemcpyAsync(devTaps, host.data(), bytes, cudaMemcpyHostToDevice, s));

  // Sequential launches on one stream preserve draw order, which matters
  // because "over" is not commutative where groups overlap in flight.
  for (size_t gi = 0; gi < groups.size(); ++gi) {
    const GroupTransform* tg = &transforms[gi * size_t(taps)];

    const RectI& b = groups[gi].bbox;
    // Same rule as the CPU path: a supplied pivot wins, so characters drawn
    // separately still turn about the word they belong to.
    const float cx = pivots ? pivots[gi * 2 + 0] : 0.5f * float(b.x1 + b.x2);
    const float cy = pivots ? pivots[gi * 2 + 1] : 0.5f * float(b.y1 + b.y2);

    // Union over all taps, so a spinning or fast-moving group is not clipped
    // to where it happens to sit at the frame's midpoint.
    RectI dest{};
    int live = 0;
    float radius = 0.0f;
    for (int k = 0; k < taps; ++k) {
      if (!tg[k].visible || tg[k].opacity <= 0.0f) continue;
      ++live;
      radius += tg[k].blur;
      dest.unionWith(tapBounds(tg[k], b, cx, cy));
    }
    if (live == 0) continue;
    radius /= float(live);

    dest.grow(2);
    dest = intersect(dest, win);
    if (dest.empty()) continue;

    const int tw = dest.width(), th = dest.height();
    const size_t tileBytes = size_t(tw) * size_t(th) * 4 * sizeof(float);
    float* tileA = static_cast<float*>(scratch.tileBuffer(0, tileBytes));
    if (!tileA) return false;

    renderTileKernel<<<gridFor(tw, th, block), block, 0, s>>>(
        tileA, srcDev, srcStride, devSeg.labels(), devSeg.labelToGroup(), devSeg.width(), width,
        height, int(gi), cx, cy, devTaps + gi * size_t(taps), taps, dest.x1, dest.y1, tw, th);

    if (radius > 0.5f) {
      float* tileB = static_cast<float*>(scratch.tileBuffer(1, tileBytes));
      if (!tileB) return false;
      // Same three passes, same radius rule, same order as the CPU path.
      const int r = std::max(1, int(radius / 3.0f + 0.5f));
      const int lineBlock = 64;
      for (int pass = 0; pass < 3; ++pass) {
        boxPassKernel<<<(th + lineBlock - 1) / lineBlock, lineBlock, 0, s>>>(tileA, tileB, tw, th,
                                                                            r, 1);
        boxPassKernel<<<(tw + lineBlock - 1) / lineBlock, lineBlock, 0, s>>>(tileB, tileA, tw, th,
                                                                            r, 0);
      }
    }

    blendTileKernel<<<gridFor(tw, th, block), block, 0, s>>>(dstDev, dstStride, tileA, dest.x1,
                                                             dest.y1, tw, th);
  }

  RTA_CUDA_OK(cudaGetLastError());

  // host[] must outlive the async upload.
  RTA_CUDA_OK(cudaStreamSynchronize(s));
  return true;
}

bool cudaDrawDiagnostics(float* dstDev, std::ptrdiff_t dstStride, const float* srcDev,
                         std::ptrdiff_t srcStride, int width, int height,
                         const std::vector<Group>& groups, const RectI& window, int lineWidth,
                         float slantTan, int yMid, void* stream) {
  if (!dstDev || !srcDev) return false;

  const RectI win = intersect(window, RectI{0, 0, width, height});
  if (win.empty()) return true;

  cudaStream_t s = asStream(stream);
  const dim3 block(16, 16);

  dimKernel<<<gridFor(win.width(), win.height(), block), block, 0, s>>>(
      dstDev, dstStride, srcDev, srcStride, win.x1, win.y1, win.width(), win.height(), 0.25f);

  const int t = std::max(1, lineWidth);
  for (size_t gi = 0; gi < groups.size(); ++gi) {
    const Group& g = groups[gi];
    const RectI& b = g.bbox;
    const bool sheared = g.sx2 > g.sx1;
    const float sx1 = sheared ? g.sx1 : float(b.x1);
    const float sx2 = sheared ? g.sx2 : float(b.x2);
    const float tanUsed = sheared ? slantTan : 0.0f;

    const int h = b.height() + 2 * t;
    // The lean moves the span sideways row by row, so the launch has to be wide
    // enough for the extremes, not just for the upright box.
    const int pad = int(std::fabs(tanUsed) * float(h)) + 2;
    const int w = int(sx2 - sx1) + 2 * t + 2 * pad;
    if (w <= 0 || h <= 0) continue;
    boxKernel<<<gridFor(w, h, block), block, 0, s>>>(dstDev, dstStride, width, height, sx1, sx2,
                                                     tanUsed, yMid, b.y1, b.y2, t, int(gi), pad,
                                                     win.x1, win.y1, win.x2, win.y2);

    // The character index this unit starts at, matching the CPU path exactly:
    // the number is what a manual override names.
    const int label = g.glyphIndex;
    const int scale = std::max(2, b.height() / 14);
    int digits = 1;
    for (int v = label; v >= 10; v /= 10) ++digits;
    const float shearTop = tanUsed * float(b.y1 - yMid);
    const int ox = int(sx1 + shearTop) + scale;
    const int oy = b.y1 - 5 * scale - 2 * scale;
    const int lw = digits * (4 * scale) - scale + 2 * scale, lh = 5 * scale + 2 * scale;
    indexKernel<<<gridFor(lw, lh, block), block, 0, s>>>(dstDev, dstStride, width, height, ox, oy,
                                                         label, digits, scale, int(gi), win.x1,
                                                         win.y1, win.x2, win.y2);
  }

  RTA_CUDA_OK(cudaGetLastError());
  return true;
}

}  // namespace rta

#endif  // RTA_WITH_CUDA
