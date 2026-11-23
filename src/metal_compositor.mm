// Metal render path. See metal_compositor.h for the why; this file is the how.
//
// Built with ARC. Every entry point runs inside its own @autoreleasepool: the
// host calls render on long-lived threads of its own that have no pool, so the
// command buffers and encoders Metal hands back autoreleased would otherwise
// pile up for the life of the thread -- the same retention the memory write-up
// (docs/memory-retention.md) is about, one command buffer per frame.
#include "metal_compositor.h"

#if RTA_WITH_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <mutex>

#include "lru_budget.h"
#include "transform_geom.h"

namespace rta {

// ------------------------------------------------------------------ kernels
//
// Compiled at runtime from this source, once per device, on the first render.
// The alternative -- a .metallib built offline -- needs the full Xcode, which
// the build otherwise does not, for a compile that takes a fraction of a second.
//
// Each kernel is the CUDA one (cuda_compositor.cu) and so the CPU compositor
// (compositor.cpp) again, statement for statement. Keep the three in step: a
// change to the sampling core has to land in all of them, and tools/metaltest
// is what says whether this one still agrees with the CPU.
//
// Compiled WITHOUT fast math. Fast math would let the compiler assume no NaN,
// and the coordinate bound below relies on NaN failing both comparisons; it
// would also swap cos/sin for approximations the CPU does not use.
static const char* kKernelSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

// A region of a buffer: pixel (x, y) starts at offset + stride * y + 4 * x.
struct Img {
  long offset;
  long stride;
  int width;
  int height;
};

struct Tap {
  float offsetX, offsetY, scale, rotation, blur, opacity;
  int visible;
};

struct ClearArgs { Img dst; int x0, y0, w, h; };
struct TileArgs {
  Img src;
  int labWidth, group;
  float cx, cy;
  int nTaps, tapBase;
  int dx0, dy0, dw, dh;
};
struct BoxArgs { int w, h; float halfWidth; int horizontal; };
struct BlendArgs { Img dst; int dx0, dy0, dw, dh; };
struct HashArgs { Img src; float threshold; int pad; };

inline long px(Img i, int x, int y) { return i.offset + i.stride * long(y) + long(x) * 4; }

kernel void clearKernel(device float* dst [[buffer(0)]],
                        constant ClearArgs& a [[buffer(1)]],
                        uint2 gid [[thread_position_in_grid]]) {
  const int x = int(gid.x), y = int(gid.y);
  if (x >= a.w || y >= a.h) return;
  const long p = px(a.dst, a.x0 + x, a.y0 + y);
  dst[p] = 0.0f; dst[p + 1] = 0.0f; dst[p + 2] = 0.0f; dst[p + 3] = 0.0f;
}

// Clamp to +-2^24 so float -> int conversion is always defined. NaN fails both
// comparisons and is sent to the far side, where every tap misses.
inline float boundCoord(float v) {
  const float lim = 16777216.0f;
  if (v < lim) return v > -lim ? v : -lim;
  return lim;
}

// Label-masked bilinear fetch. Masking before filtering is what keeps
// antialiased glyph edges intact and stops a moving word dragging in a
// fragment of its neighbour.
inline float4 sampleMasked(device const float* src, Img s, device const int* labels,
                           device const int* labelToGroup, int labWidth, int group,
                           float fx, float fy) {
  float4 out = float4(0.0f);
  fx = boundCoord(fx);
  fy = boundCoord(fy);
  const int x0 = int(floor(fx));
  const int y0 = int(floor(fy));
  const float tx = fx - float(x0);
  const float ty = fy - float(y0);
  const float wx[2] = {1.0f - tx, tx};
  const float wy[2] = {1.0f - ty, ty};

  for (int j = 0; j < 2; ++j) {
    const int sy = y0 + j;
    if (sy < 0 || sy >= s.height) continue;
    for (int i = 0; i < 2; ++i) {
      const int sx = x0 + i;
      if (sx < 0 || sx >= s.width) continue;
      const int lab = labels[long(sy) * long(labWidth) + long(sx)];
      if (lab == 0 || labelToGroup[lab] != group) continue;
      const float wgt = wx[i] * wy[j];
      if (wgt <= 0.0f) continue;
      const long p = px(s, sx, sy);
      out.x += src[p] * wgt;
      out.y += src[p + 1] * wgt;
      out.z += src[p + 2] * wgt;
      out.w += src[p + 3] * wgt;
    }
  }
  return out;
}

// One thread per tile pixel; each averages every shutter tap. The group goes
// into its OWN tile, because the defocus that follows has to see the sprite
// alone -- blurring into the frame would drag in whatever is under it.
kernel void renderTileKernel(device float* tile [[buffer(0)]],
                             device const float* src [[buffer(1)]],
                             device const int* labels [[buffer(2)]],
                             device const int* labelToGroup [[buffer(3)]],
                             device const Tap* taps [[buffer(4)]],
                             constant TileArgs& a [[buffer(5)]],
                             uint2 gid [[thread_position_in_grid]]) {
  const int lx = int(gid.x), ly = int(gid.y);
  if (lx >= a.dw || ly >= a.dh) return;
  const int x = a.dx0 + lx;
  const int y = a.dy0 + ly;

  float4 acc = float4(0.0f);
  for (int k = 0; k < a.nTaps; ++k) {
    const Tap t = taps[a.tapBase + k];
    if (t.visible == 0 || t.opacity <= 0.0f) continue;

    // Inverse of  p' = centre + R(theta)*scale*(p - centre) + offset
    const float inv = 1.0f / (fabs(t.scale) < 1e-6f ? 1e-6f : t.scale);
    const float c = cos(-t.rotation) * inv;
    const float s = sin(-t.rotation) * inv;
    const float ddx = float(x) + 0.5f - a.cx - t.offsetX;
    const float ddy = float(y) + 0.5f - a.cy - t.offsetY;
    const float fx = c * ddx - s * ddy + a.cx - 0.5f;
    const float fy = s * ddx + c * ddy + a.cy - 0.5f;

    const float4 smp = sampleMasked(src, a.src, labels, labelToGroup, a.labWidth, a.group, fx, fy);
    if (smp.w <= 0.0f) continue;
    acc.x += smp.x * t.opacity;
    acc.y += smp.y * t.opacity;
    acc.z += smp.z * t.opacity;
    acc.w += smp.w * t.opacity;
  }

  const float invTaps = 1.0f / float(a.nTaps);
  const long o = (long(ly) * long(a.dw) + long(lx)) * 4;
  tile[o] = acc.x * invTaps;
  tile[o + 1] = acc.y * invTaps;
  tile[o + 2] = acc.z * invTaps;
  tile[o + 3] = acc.w * invTaps;
}

// One thread per LINE, walking it with a running window: the same additions in
// the same order as the CPU boxPass, so the two cannot drift apart.
kernel void boxPassKernel(device const float* in [[buffer(0)]],
                          device float* out [[buffer(1)]],
                          constant BoxArgs& a [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
  const int n = a.horizontal ? a.w : a.h;
  const int lines = a.horizontal ? a.h : a.w;
  const int l = int(gid);
  if (l >= lines) return;

  const long step = a.horizontal ? 4 : long(a.w) * 4;
  const long lineStep = a.horizontal ? long(a.w) * 4 : 4;

  const int r = int(a.halfWidth);
  const float f = a.halfWidth - float(r);
  const float inv = 1.0f / (2.0f * float(r) + 1.0f + 2.0f * f);
  device const float* ip = in + lineStep * l;
  device float* op = out + lineStep * l;

  float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int i = 0; i <= r && i < n; ++i)
    for (int c = 0; c < 4; ++c) sum[c] += ip[step * i + c];

  for (int i = 0; i < n; ++i) {
    const int lo = i - r - 1, hi = i + r + 1;
    for (int c = 0; c < 4; ++c) {
      float edge = 0.0f;
      if (lo >= 0) edge += ip[step * lo + c];
      if (hi < n) edge += ip[step * hi + c];
      op[step * i + c] = (sum[c] + f * edge) * inv;
    }
    const int add = i + r + 1, drop = i - r;
    if (add < n)
      for (int c = 0; c < 4; ++c) sum[c] += ip[step * add + c];
    if (drop >= 0)
      for (int c = 0; c < 4; ++c) sum[c] -= ip[step * drop + c];
  }
}

// Premultiplied "over" of the finished tile onto the frame.
kernel void blendTileKernel(device float* dst [[buffer(0)]],
                            device const float* tile [[buffer(1)]],
                            constant BlendArgs& a [[buffer(2)]],
                            uint2 gid [[thread_position_in_grid]]) {
  const int lx = int(gid.x), ly = int(gid.y);
  if (lx >= a.dw || ly >= a.dh) return;
  const long t = (long(ly) * long(a.dw) + long(lx)) * 4;
  const float al = tile[t + 3];
  if (al <= 0.0f) return;
  const long o = px(a.dst, a.dx0 + lx, a.dy0 + ly);
  const float ia = 1.0f - al;
  dst[o] = tile[t] + dst[o] * ia;
  dst[o + 1] = tile[t + 1] + dst[o + 1] * ia;
  dst[o + 2] = tile[t + 2] + dst[o + 2] * ia;
  dst[o + 3] = al + dst[o + 3] * ia;
}

// Every 8th pixel in both axes, like the CPU and CUDA hashes. Metal has no
// 64-bit atomic add, so the sum is kept as two 32-bit words with the carry out
// of the low one added to the high one. Each add to the low word is atomic, so
// the carries counted come to exactly the carries of the total, whatever the
// order -- the result is the 64-bit sum CUDA's atomicAdd gives.
kernel void hashKernel(device const float* src [[buffer(0)]],
                       device atomic_uint* acc [[buffer(1)]],
                       constant HashArgs& a [[buffer(2)]],
                       uint2 gid [[thread_position_in_grid]]) {
  const int x = int(gid.x) * 8, y = int(gid.y) * 8;
  if (x >= a.src.width || y >= a.src.height) return;

  const float al = src[px(a.src, x, y) + 3];
  const uint q = uint(min(15.0f, max(0.0f, al) * 15.0f));
  ulong h = ulong(q + (al > a.threshold ? 16u : 0u));
  // Mix in the position so a moved word changes the hash even if the histogram
  // of alpha values does not.
  h ^= ulong(x) * 0x9E3779B97F4A7C15ul;
  h ^= ulong(y) * 0xC2B2AE3D27D4EB4Ful;
  h *= 0xFF51AFD7ED558CCDul;
  h ^= h >> 33;

  const uint lo = uint(h & 0xFFFFFFFFul);
  const uint hi = uint(h >> 32);
  const uint old = atomic_fetch_add_explicit(&acc[0], lo, memory_order_relaxed);
  const uint carry = (old + lo < old) ? 1u : 0u;
  atomic_fetch_add_explicit(&acc[1], hi + carry, memory_order_relaxed);
}
)MSL";

namespace {

// Host mirrors of the kernel argument structs. Laid out by hand to match MSL:
// `long` is 8 bytes and 8-aligned there as int64_t is here.
struct Img {
  int64_t offset;
  int64_t stride;
  int32_t width;
  int32_t height;
};
struct DevTap {
  float offsetX, offsetY, scale, rotation, blur, opacity;
  int32_t visible;
};
struct ClearArgs {
  Img dst;
  int32_t x0, y0, w, h;
};
struct TileArgs {
  Img src;
  int32_t labWidth, group;
  float cx, cy;
  int32_t nTaps, tapBase;
  int32_t dx0, dy0, dw, dh;
};
struct BoxArgs {
  int32_t w, h;
  float halfWidth;
  int32_t horizontal;
};
struct BlendArgs {
  Img dst;
  int32_t dx0, dy0, dw, dh;
};
struct HashArgs {
  Img src;
  float threshold;
  int32_t pad;
};
static_assert(sizeof(Img) == 24, "Img must match the MSL struct");
static_assert(sizeof(DevTap) == 28, "DevTap must match the MSL struct");
static_assert(sizeof(ClearArgs) == 40, "ClearArgs must match the MSL struct");
static_assert(sizeof(TileArgs) == 64, "TileArgs must match the MSL struct");
static_assert(sizeof(BoxArgs) == 16, "BoxArgs must match the MSL struct");
static_assert(sizeof(BlendArgs) == 40, "BlendArgs must match the MSL struct");
static_assert(sizeof(HashArgs) == 32, "HashArgs must match the MSL struct");

Img toImg(const MetalImage& m) {
  return Img{m.offset, int64_t(m.rowStride), int32_t(m.width), int32_t(m.height)};
}

inline id<MTLCommandQueue> asQueue(void* q) { return (__bridge id<MTLCommandQueue>)q; }
inline id<MTLBuffer> asBuffer(void* b) { return (__bridge id<MTLBuffer>)b; }

RectI intersect(const RectI& a, const RectI& b) {
  RectI r{std::max(a.x1, b.x1), std::max(a.y1, b.y1), std::min(a.x2, b.x2), std::min(a.y2, b.y2)};
  return r.empty() ? RectI{0, 0, 0, 0} : r;
}

double nowSeconds() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------- pipelines

struct Pipelines {
  id<MTLComputePipelineState> clear, tile, box, blend, hash;
};

std::mutex& pipelineMutex() {
  static std::mutex m;
  return m;
}

// Per device, by registry ID. A device whose compile failed is remembered as
// null, so a broken shader costs one attempt rather than one per frame. Never
// destroyed, for the same reason as the store below.
std::map<uint64_t, std::shared_ptr<Pipelines>>& pipelineCache() {
  static auto* m = new std::map<uint64_t, std::shared_ptr<Pipelines>>();
  return *m;
}

std::shared_ptr<Pipelines> compilePipelines(id<MTLDevice> dev) {
  MTLCompileOptions* opts = [MTLCompileOptions new];
  if (@available(macOS 15.0, *)) {
    opts.mathMode = MTLMathModeSafe;
  } else {
    opts.fastMathEnabled = NO;
  }
  NSError* err = nil;
  id<MTLLibrary> lib = [dev newLibraryWithSource:@(kKernelSource) options:opts error:&err];
  if (!lib) {
    NSLog(@"TextAnimator: Metal kernels failed to compile: %@", err);
    return nullptr;
  }
  auto make = [&](const char* name) -> id<MTLComputePipelineState> {
    id<MTLFunction> fn = [lib newFunctionWithName:@(name)];
    if (!fn) return nil;
    NSError* e = nil;
    id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:fn error:&e];
    if (!ps) NSLog(@"TextAnimator: Metal pipeline %s failed: %@", name, e);
    return ps;
  };
  auto p = std::make_shared<Pipelines>();
  p->clear = make("clearKernel");
  p->tile = make("renderTileKernel");
  p->box = make("boxPassKernel");
  p->blend = make("blendTileKernel");
  p->hash = make("hashKernel");
  if (!p->clear || !p->tile || !p->box || !p->blend || !p->hash) return nullptr;
  return p;
}

// Compiles on first use. The lock is held across the compile on purpose -- two
// renders arriving together should not both compile -- and that is safe because
// compiling is CPU work: it never waits on a command queue, so it cannot be
// caught up in the host's GPU work the way the store's lock once was.
std::shared_ptr<Pipelines> pipelinesFor(id<MTLDevice> dev) {
  if (!dev) return nullptr;
  std::lock_guard<std::mutex> lock(pipelineMutex());
  auto& cache = pipelineCache();
  const uint64_t key = dev.registryID;
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  std::shared_ptr<Pipelines> p = compilePipelines(dev);
  cache[key] = p;
  return p;
}

// Thread shapes from the pipeline rather than fixed, because a kernel heavy on
// registers can be limited below 256 threads per group on some GPUs. Whole
// threadgroups, not dispatchThreads, so no GPU family question arises; every
// kernel bounds-checks its own edge.
void dispatch2D(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> ps, int w, int h) {
  if (w <= 0 || h <= 0) return;
  const NSUInteger tx = std::max<NSUInteger>(1, ps.threadExecutionWidth);
  const NSUInteger ty =
      std::max<NSUInteger>(1, std::min<NSUInteger>(16, ps.maxTotalThreadsPerThreadgroup / tx));
  [enc setComputePipelineState:ps];
  [enc dispatchThreadgroups:MTLSizeMake((NSUInteger(w) + tx - 1) / tx, (NSUInteger(h) + ty - 1) / ty, 1)
      threadsPerThreadgroup:MTLSizeMake(tx, ty, 1)];
}

void dispatch1D(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> ps, int n) {
  if (n <= 0) return;
  const NSUInteger t = std::max<NSUInteger>(1, std::min<NSUInteger>(64, ps.maxTotalThreadsPerThreadgroup));
  [enc setComputePipelineState:ps];
  [enc dispatchThreadgroups:MTLSizeMake((NSUInteger(n) + t - 1) / t, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(t, 1, 1)];
}

void encodeClear(id<MTLComputeCommandEncoder> enc, const Pipelines& p, id<MTLBuffer> dst,
                 const MetalImage& img, const RectI& win) {
  ClearArgs a{toImg(img), win.x1, win.y1, win.width(), win.height()};
  [enc setBuffer:dst offset:0 atIndex:0];
  [enc setBytes:&a length:sizeof(a) atIndex:1];
  dispatch2D(enc, p.clear, win.width(), win.height());
}

// Shared on Apple silicon, where the GPU reads system memory directly; managed
// on a Mac with its own video memory, so the GPU reads a copy in VRAM rather
// than every label lookup crossing the bus.
MTLResourceOptions uploadStorage(id<MTLDevice> dev) {
  if (dev.hasUnifiedMemory) return MTLResourceStorageModeShared;
#if TARGET_OS_OSX
  return MTLResourceStorageModeManaged;
#else
  return MTLResourceStorageModeShared;
#endif
}

}  // namespace

// ------------------------------------------------------------------ stencil

class MetalStencil {
 public:
  id<MTLBuffer> labels = nil;
  id<MTLBuffer> labelToGroup = nil;
  int width = 0, height = 0;
  size_t bytes() const {
    return (labels ? labels.length : 0) + (labelToGroup ? labelToGroup.length : 0);
  }
};

namespace {

// A new stencil per upload rather than one rewritten in place: a frame still in
// flight may be reading the old buffers, and Metal keeps those alive until it
// is done with them -- so replacing never has to wait for anything.
std::shared_ptr<MetalStencil> uploadStencil(id<MTLDevice> dev, const Segmentation& seg) {
  if (seg.labelImage.empty() || seg.labelToGroup.empty()) return nullptr;
  static_assert(sizeof(seg.labelImage[0]) == 4 && sizeof(seg.labelToGroup[0]) == 4,
                "the kernels read 32-bit labels");
  auto s = std::make_shared<MetalStencil>();
  const MTLResourceOptions opts = uploadStorage(dev);
  s->labels = [dev newBufferWithBytes:seg.labelImage.data()
                               length:seg.labelImage.size() * sizeof(seg.labelImage[0])
                              options:opts];
  s->labelToGroup = [dev newBufferWithBytes:seg.labelToGroup.data()
                                     length:seg.labelToGroup.size() * sizeof(seg.labelToGroup[0])
                                    options:opts];
  if (!s->labels || !s->labelToGroup) return nullptr;
  s->width = seg.width;
  s->height = seg.height;
  return s;
}

// ------------------------------------------------- device memory, shared
// The same arrangement as the CUDA store, for the same reason (see
// docs/memory-retention.md): the host keeps an instance per use of the effect,
// so what is held has to be bounded by what renders at once, not by instances.

constexpr size_t kStencilBudget = size_t(256) << 20;
constexpr double kRecentSeconds = 3.0;
constexpr double kIdleScratchSeconds = 10.0;

struct StencilEntry {
  std::shared_ptr<MetalStencil> dev;
  std::weak_ptr<const Segmentation> src;  // weak: a freed one cannot match a new one at its address
};

struct DeviceStore {
  std::vector<MetalLease::Slot*> idle;
  std::map<const void*, StencilEntry> stencils;
  std::vector<const void*> forgotten;
  LruBudget budget;
};

// Guards bookkeeping only. Nothing that waits on the GPU happens under it --
// the rule the CUDA store learned the hard way. Releasing a Metal buffer never
// waits (work in flight keeps its own reference), but stencils are still let go
// after the lock, so the rule holds without having to think about it.
std::mutex& storeMutex() {
  static std::mutex m;
  return m;
}

std::atomic<size_t> gStencilBytes{0};
std::atomic<size_t> gScratchBytes{0};
std::atomic<int> gScratchCount{0};

// Never destroyed: the host may unload the plugin after Metal has begun tearing
// down, and the process ending returns everything anyway.
std::map<uint64_t, DeviceStore>& stores() {
  static auto* s = new std::map<uint64_t, DeviceStore>();
  return *s;
}

using Released = std::vector<std::shared_ptr<MetalStencil>>;

void dropStencil(DeviceStore& store, const void* owner, Released& out) {
  auto it = store.stencils.find(owner);
  if (it != store.stencils.end()) {
    out.push_back(std::move(it->second.dev));
    store.stencils.erase(it);
  }
  store.budget.remove(owner);
}

void publishStencilBytes() {
  size_t t = 0;
  for (const auto& kv : stores()) t += kv.second.budget.total();
  gStencilBytes.store(t);
}

}  // namespace

// A slot belongs to exactly one render -- and then to that render's GPU work --
// whenever it is out of the idle list, so its fields need no lock.
struct MetalLease::Slot {
  id<MTLBuffer> tiles[2] = {nil, nil};
  size_t peak[2] = {0, 0};  // largest asked of each tile since the last trim
  double lastUse = 0.0;
  size_t heldBytes = 0;     // what it held when last returned, for the readout

  size_t bytes() const {
    return (tiles[0] ? tiles[0].length : 0) + (tiles[1] ? tiles[1].length : 0);
  }

  // Grow-only between trims, rounded up a quarter so groups of slightly
  // different sizes do not reallocate one after another. Private storage: only
  // the GPU ever touches a tile. Replacing one mid-frame is safe, because the
  // command buffer keeps the old buffer alive for the dispatches already
  // encoded against it.
  id<MTLBuffer> tile(id<MTLDevice> dev, int which, size_t bytes) {
    peak[which] = std::max(peak[which], bytes);
    if (tiles[which] && tiles[which].length >= bytes) return tiles[which];
    tiles[which] = [dev newBufferWithLength:bytes + bytes / 4 options:MTLResourceStorageModePrivate];
    return tiles[which];
  }

  // Same policy as CudaScratch::trim: give back a large tile far bigger than
  // anything asked for since the last trim.
  void trim() {
    constexpr size_t kWorthFreeing = size_t(64) << 20;
    for (int i = 0; i < 2; ++i) {
      if (tiles[i] && tiles[i].length > kWorthFreeing && tiles[i].length > peak[i] * 4)
        tiles[i] = nil;
      peak[i] = 0;
    }
  }

  void releaseTiles() {
    tiles[0] = tiles[1] = nil;
    peak[0] = peak[1] = 0;
  }
};

namespace {

void setHeld(MetalLease::Slot* s, size_t bytes) {
  gScratchBytes.fetch_add(bytes);
  gScratchBytes.fetch_sub(s->heldBytes);
  s->heldBytes = bytes;
}

// Back to the idle list. Called once the GPU has finished with the slot's
// tiles -- from a command buffer's completion handler, or straight away when
// no work was ever committed against them.
void returnSlot(MetalLease::Slot* slot, uint64_t device) {
  slot->trim();
  slot->lastUse = nowSeconds();
  setHeld(slot, slot->bytes());
  std::lock_guard<std::mutex> lock(storeMutex());
  stores()[device].idle.push_back(slot);
}

}  // namespace

MetalLease::MetalLease(void* queue) {
  @autoreleasepool {
    id<MTLCommandQueue> q = asQueue(queue);
    if (!q) return;
    device_ = q.device.registryID;

    // A scratch only exists because renders once overlapped. Those that have
    // sat idle a while give their tiles back. A slot in the idle list has no
    // GPU work in flight -- it only gets there once its work has finished --
    // so letting its buffers go is immediate and safe.
    std::vector<Slot*> stale;
    {
      std::lock_guard<std::mutex> lock(storeMutex());
      std::vector<Slot*>& idle = stores()[device_].idle;
      if (!idle.empty()) {
        slot_ = idle.back();
        idle.pop_back();
      }
      const double now = nowSeconds();
      for (auto it = idle.begin(); it != idle.end();) {
        if ((*it)->heldBytes > 0 && now - (*it)->lastUse > kIdleScratchSeconds) {
          stale.push_back(*it);
          it = idle.erase(it);
        } else {
          ++it;
        }
      }
    }
    if (!stale.empty()) {
      for (Slot* s : stale) {
        s->releaseTiles();
        setHeld(s, 0);
      }
      std::lock_guard<std::mutex> lock(storeMutex());
      std::vector<Slot*>& idle = stores()[device_].idle;
      idle.insert(idle.end(), stale.begin(), stale.end());
    }
    if (!slot_) {
      slot_ = new Slot();
      gScratchCount.fetch_add(1);
    }
  }
}

MetalLease::~MetalLease() {
  if (!slot_ || handedOff_) return;
  @autoreleasepool {
    returnSlot(slot_, device_);
  }
}

// ------------------------------------------------------------- entry points

bool metalAvailable() {
  static const bool ok = [] {
    @autoreleasepool {
      id<MTLDevice> d = MTLCreateSystemDefaultDevice();
      return d != nil;
    }
  }();
  return ok;
}

std::shared_ptr<const MetalStencil> metalStencil(const void* owner,
                                                 const std::shared_ptr<const Segmentation>& seg,
                                                 void* queue) {
  if (!owner || !seg || !queue) return nullptr;
  @autoreleasepool {
    id<MTLDevice> dev = asQueue(queue).device;
    const uint64_t device = dev.registryID;
    // Declared first so it is destroyed last, after every lock is let go.
    Released released;
    {
      std::lock_guard<std::mutex> lock(storeMutex());
      DeviceStore& store = stores()[device];
      for (const void* gone : store.forgotten) dropStencil(store, gone, released);
      store.forgotten.clear();

      auto it = store.stencils.find(owner);
      if (it != store.stencils.end() && it->second.dev && it->second.src.lock() == seg) {
        // Already held: every frame of a steady title.
        std::shared_ptr<const MetalStencil> out = it->second.dev;
        const double now = nowSeconds();
        store.budget.touch(owner, it->second.dev->bytes(), now);
        for (const void* v : store.budget.victims(kStencilBudget, now, kRecentSeconds, owner))
          dropStencil(store, v, released);
        publishStencilBytes();
        return out;
      }
    }

    std::shared_ptr<MetalStencil> fresh = uploadStencil(dev, *seg);

    std::lock_guard<std::mutex> lock(storeMutex());
    DeviceStore& store = stores()[device];
    if (!fresh) {
      dropStencil(store, owner, released);
      publishStencilBytes();
      return nullptr;
    }
    StencilEntry& e = store.stencils[owner];
    if (e.dev) released.push_back(std::move(e.dev));
    e.dev = fresh;
    e.src = seg;
    const double now = nowSeconds();
    store.budget.touch(owner, fresh->bytes(), now);
    for (const void* v : store.budget.victims(kStencilBudget, now, kRecentSeconds, owner))
      dropStencil(store, v, released);
    publishStencilBytes();
    return fresh;
  }
}

void metalForgetStencil(const void* owner) {
  std::lock_guard<std::mutex> lock(storeMutex());
  for (auto& kv : stores())
    if (kv.second.stencils.count(owner)) kv.second.forgotten.push_back(owner);
}

DeviceMemoryUse deviceMemoryUse() {
  DeviceMemoryUse u;
  u.stencils = gStencilBytes.load();
  u.scratch = gScratchBytes.load();
  u.scratchCount = gScratchCount.load();
  return u;
}

uint64_t metalHashAlpha(const MetalImage& src, float threshold, void* queue, bool* ok) {
  if (ok) *ok = false;
  if (!src.valid() || !queue) return 0;
  @autoreleasepool {
    id<MTLCommandQueue> q = asQueue(queue);
    std::shared_ptr<Pipelines> p = pipelinesFor(q.device);
    if (!p) return 0;

    id<MTLBuffer> acc = [q.device newBufferWithLength:8 options:MTLResourceStorageModeShared];
    if (!acc) return 0;
    std::memset(acc.contents, 0, 8);

    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return 0;
    HashArgs a{toImg(src), threshold, 0};
    [enc setBuffer:asBuffer(src.buffer) offset:0 atIndex:0];
    [enc setBuffer:acc offset:0 atIndex:1];
    [enc setBytes:&a length:sizeof(a) atIndex:2];
    dispatch2D(enc, p->hash, (src.width + 7) / 8, (src.height + 7) / 8);
    [enc endEncoding];
    [cmd commit];
    // The one wait per frame. The hash decides whether to analyse again, so
    // the CPU cannot go on without it -- CUDA synchronises its stream here too.
    [cmd waitUntilCompleted];
    if (cmd.status != MTLCommandBufferStatusCompleted) return 0;

    const uint32_t* words = static_cast<const uint32_t*>(acc.contents);
    uint64_t host = uint64_t(words[0]) | (uint64_t(words[1]) << 32);
    host ^= (uint64_t(src.width) << 32) ^ uint64_t(src.height);
    if (ok) *ok = true;
    return host;
  }
}

bool metalDownload(std::vector<float>& out, const MetalImage& src, void* queue) {
  if (!src.valid() || !queue) return false;
  @autoreleasepool {
    id<MTLCommandQueue> q = asQueue(queue);
    // The region's rows from the one with the lowest address, |stride| apart --
    // the layout cudaDownload produces, so the caller treats both alike.
    const int64_t absStride = src.rowStride < 0 ? -int64_t(src.rowStride) : int64_t(src.rowStride);
    const int64_t first =
        src.rowStride < 0 ? src.offset + int64_t(src.rowStride) * (src.height - 1) : src.offset;
    if (first < 0) return false;
    const size_t floats = size_t(absStride) * size_t(src.height - 1) + size_t(src.width) * 4;
    const size_t bytes = floats * sizeof(float);
    id<MTLBuffer> buf = asBuffer(src.buffer);
    if (size_t(first) * sizeof(float) + bytes > buf.length) return false;

    // Copied on the GPU into a buffer the CPU can read, rather than read in
    // place: the host's buffer may well be private to the GPU.
    id<MTLBuffer> staging = [q.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!staging) return false;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    if (!cmd || !blit) return false;
    [blit copyFromBuffer:buf
            sourceOffset:NSUInteger(first) * sizeof(float)
                toBuffer:staging
       destinationOffset:0
                    size:bytes];
    [blit endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.status != MTLCommandBufferStatusCompleted) return false;

    out.assign(size_t(absStride) * size_t(src.height), 0.0f);
    std::memcpy(out.data(), staging.contents, bytes);
    return true;
  }
}

bool metalClearWindow(const MetalImage& dst, const RectI& window, void* queue) {
  if (!dst.valid() || !queue) return false;
  const RectI win = intersect(window, RectI{0, 0, dst.width, dst.height});
  if (win.empty()) return true;
  @autoreleasepool {
    id<MTLCommandQueue> q = asQueue(queue);
    std::shared_ptr<Pipelines> p = pipelinesFor(q.device);
    if (!p) return false;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return false;
    encodeClear(enc, *p, asBuffer(dst.buffer), dst, win);
    [enc endEncoding];
    [cmd commit];
    return true;
  }
}

bool metalCompositeGroups(const MetalImage& dst, const MetalImage& src,
                          const MetalStencil& stencil, const std::vector<Group>& groups,
                          const std::vector<GroupTransform>& transforms, int taps,
                          const RectI& window, MetalLease& lease, void* queue,
                          const float* pivots) {
  taps = std::max(1, taps);
  if (!dst.valid() || !src.valid() || !queue || !lease.valid() || !stencil.labels ||
      transforms.size() != groups.size() * size_t(taps))
    return false;

  const RectI win = intersect(window, RectI{0, 0, dst.width, dst.height});
  if (win.empty()) return true;

  @autoreleasepool {
    id<MTLCommandQueue> q = asQueue(queue);
    id<MTLDevice> dev = q.device;
    std::shared_ptr<Pipelines> p = pipelinesFor(dev);
    if (!p) return false;

    // Every tap in one upload. A fresh buffer per frame, so nothing has to wait
    // for the last frame to finish reading the previous one.
    std::vector<DevTap> host(transforms.size());
    for (size_t i = 0; i < transforms.size(); ++i) {
      const GroupTransform& t = transforms[i];
      host[i] = DevTap{t.offsetX, t.offsetY, t.scale, t.rotation, t.blur, t.opacity, t.visible ? 1 : 0};
    }
    id<MTLBuffer> tapBuf = [dev newBufferWithBytes:host.data()
                                            length:std::max<size_t>(sizeof(DevTap), host.size() * sizeof(DevTap))
                                           options:MTLResourceStorageModeShared];
    if (!tapBuf) return false;

    id<MTLBuffer> dstBuf = asBuffer(dst.buffer);
    id<MTLBuffer> srcBuf = asBuffer(src.buffer);

    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!cmd || !enc) return false;

    // One encoder, serial dispatch: each dispatch sees everything the previous
    // ones wrote. That is what makes reusing one tile across groups safe, and
    // it keeps draw order, which matters because "over" is not commutative
    // where groups overlap in flight.
    encodeClear(enc, *p, dstBuf, dst, win);

    MetalLease::Slot* slot = lease.slot_;
    bool good = true;
    for (size_t gi = 0; gi < groups.size() && good; ++gi) {
      const GroupTransform* tg = &transforms[gi * size_t(taps)];
      const RectI& b = groups[gi].bbox;
      // A supplied pivot wins, so characters drawn separately still turn about
      // the word they belong to.
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
      id<MTLBuffer> tileA = slot->tile(dev, 0, tileBytes);
      if (!tileA) {
        good = false;
        break;
      }

      TileArgs ta{toImg(src), int32_t(stencil.width), int32_t(gi), cx, cy, int32_t(taps),
                  int32_t(gi * size_t(taps)), dest.x1, dest.y1, tw, th};
      [enc setBuffer:tileA offset:0 atIndex:0];
      [enc setBuffer:srcBuf offset:0 atIndex:1];
      [enc setBuffer:stencil.labels offset:0 atIndex:2];
      [enc setBuffer:stencil.labelToGroup offset:0 atIndex:3];
      [enc setBuffer:tapBuf offset:0 atIndex:4];
      [enc setBytes:&ta length:sizeof(ta) atIndex:5];
      dispatch2D(enc, p->tile, tw, th);

      // Only skipped where the window is narrow enough to BE the identity.
      if (radius > 0.003f) {
        id<MTLBuffer> tileB = slot->tile(dev, 1, tileBytes);
        if (!tileB) {
          good = false;
          break;
        }
        // Same three passes, same width, same order as the CPU path.
        const float halfWidth = radius / 3.0f;
        const BoxArgs horiz{tw, th, halfWidth, 1};
        const BoxArgs vert{tw, th, halfWidth, 0};
        for (int pass = 0; pass < 3; ++pass) {
          [enc setBuffer:tileA offset:0 atIndex:0];
          [enc setBuffer:tileB offset:0 atIndex:1];
          [enc setBytes:&horiz length:sizeof(horiz) atIndex:2];
          dispatch1D(enc, p->box, th);
          [enc setBuffer:tileB offset:0 atIndex:0];
          [enc setBuffer:tileA offset:0 atIndex:1];
          [enc setBytes:&vert length:sizeof(vert) atIndex:2];
          dispatch1D(enc, p->box, tw);
        }
      }

      BlendArgs ba{toImg(dst), dest.x1, dest.y1, tw, th};
      [enc setBuffer:dstBuf offset:0 atIndex:0];
      [enc setBuffer:tileA offset:0 atIndex:1];
      [enc setBytes:&ba length:sizeof(ba) atIndex:2];
      dispatch2D(enc, p->blend, tw, th);
    }
    [enc endEncoding];
    if (!good) return false;  // never committed: dst keeps the earlier clear

    // The tiles go back to the store when the GPU is done with them, not now.
    // Nothing on the CPU waits for the composite: the host reads dst from the
    // same queue, which runs this first.
    const uint64_t device = lease.device_;
    [cmd addCompletedHandler:^(id<MTLCommandBuffer>) {
      @autoreleasepool {
        returnSlot(slot, device);
      }
    }];
    lease.handedOff_ = true;
    [cmd commit];
    return true;
  }
}

void metalFinish(void* queue) {
  if (!queue) return;
  @autoreleasepool {
    id<MTLCommandBuffer> cmd = [asQueue(queue) commandBuffer];
    [cmd commit];
    [cmd waitUntilCompleted];
  }
}

void* metalCreateQueue() {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) return nullptr;
    id<MTLCommandQueue> q = [dev newCommandQueue];
    return q ? (__bridge_retained void*)q : nullptr;
  }
}

void* metalCreateBuffer(void* queue, size_t bytes, const void* data) {
  if (!queue || bytes == 0) return nullptr;
  @autoreleasepool {
    id<MTLDevice> dev = asQueue(queue).device;
    id<MTLBuffer> b = data ? [dev newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared]
                           : [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    return b ? (__bridge_retained void*)b : nullptr;
  }
}

const void* metalBufferContents(void* buffer) {
  return buffer ? asBuffer(buffer).contents : nullptr;
}

}  // namespace rta

#endif  // RTA_WITH_METAL
