// CPU vs Metal compositor parity -- the counterpart of cudatest.
//
// The CPU path is the one segtest exercises, so it is the reference; the Metal
// kernels are a third implementation of the same maths, and on a Mac every
// frame Resolve draws goes through them. Early frames of a reveal are where they
// would diverge: large slide offsets, low opacity, groups not started yet.
//
// The images are laid out the way an OFX host hands them over -- bottom-up rows,
// padded, starting part-way into the buffer -- so the addressing the plugin
// relies on is under test too, not just the arithmetic. The fingerprint and the
// readback used on a segmentation miss are checked as well.
//
//   metaltest <in.png> [--mode char|word|line] [--frames N] [--slide PX]
//             [--scale F] [--rotate DEG] [--blur PX] [--mblur 0|1] [--track F]
//             [--linespace F]
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "animator.h"
#include "compositor.h"
#include "metal_compositor.h"
#include "segmentation.h"

namespace {

// Rows bottom-up, `pad` floats past each row, and `lead` floats of something
// else before the image starts -- as a host's buffer may be.
struct HostLayout {
  int w = 0, h = 0;
  std::ptrdiff_t stride = 0;  // floats per stored row
  int64_t lead = 0;           // floats before the bottom row

  size_t floats() const { return size_t(lead) + size_t(stride) * size_t(h); }
  // The top-down view the plugin builds: (0,0) is the top row, stride negative.
  rta::MetalImage view(void* buffer) const {
    rta::MetalImage m;
    m.buffer = buffer;
    m.width = w;
    m.height = h;
    m.rowStride = -stride;
    m.offset = lead + int64_t(stride) * (h - 1);
    return m;
  }
  size_t index(int x, int yTopDown) const {
    return size_t(lead) + size_t(stride) * size_t(h - 1 - yTopDown) + size_t(x) * 4;
  }
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: metaltest <in.png> [--mode char|word|line] [--frames N] [--slide PX]\n"
                 "                 [--scale F] [--blur PX] [--mblur 0|1] [--track F] "
                 "[--linespace F]\n");
    return 2;
  }
  rta::DetectParams p;
  rta::SpacingParams sp;
  int frameCount = 24;
  double startBlur = 0.0;
  bool motionBlur = false;
  double slide = 45.0;
  double startScale = 1.0;
  // Rotation is the one place the GPU's maths library is in play at all --
  // cos and sin of the tap angle -- so it needs a run of its own.
  double startRotation = 0.0;
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--mode") && i + 1 < argc) {
      const char* m = argv[++i];
      p.mode = !std::strcmp(m, "char")   ? rta::GroupMode::Character
               : !std::strcmp(m, "line") ? rta::GroupMode::Line
                                         : rta::GroupMode::Word;
    } else if (!std::strcmp(argv[i], "--frames") && i + 1 < argc) {
      frameCount = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--slide") && i + 1 < argc) {
      slide = std::atof(argv[++i]);
    } else if (!std::strcmp(argv[i], "--rotate") && i + 1 < argc) {
      startRotation = std::atof(argv[++i]);
    } else if (!std::strcmp(argv[i], "--scale") && i + 1 < argc) {
      startScale = std::atof(argv[++i]);
    } else if (!std::strcmp(argv[i], "--blur") && i + 1 < argc) {
      startBlur = std::atof(argv[++i]);
    } else if (!std::strcmp(argv[i], "--mblur") && i + 1 < argc) {
      motionBlur = std::atoi(argv[++i]) != 0;
    } else if (!std::strcmp(argv[i], "--track") && i + 1 < argc) {
      sp.tracking = float(std::atof(argv[++i]));
    } else if (!std::strcmp(argv[i], "--linespace") && i + 1 < argc) {
      sp.lineSpacing = float(std::atof(argv[++i]));
    }
  }

  int w = 0, h = 0, n = 0;
  unsigned char* png = stbi_load(argv[1], &w, &h, &n, 4);
  if (!png) {
    std::fprintf(stderr, "cannot read %s\n", argv[1]);
    return 1;
  }
  std::vector<float> src(size_t(w) * h * 4);
  for (size_t i = 0; i < size_t(w) * h; ++i) {
    const float a = png[i * 4 + 3] / 255.0f;
    src[i * 4 + 0] = (png[i * 4 + 0] / 255.0f) * a;
    src[i * 4 + 1] = (png[i * 4 + 1] / 255.0f) * a;
    src[i * 4 + 2] = (png[i * 4 + 2] / 255.0f) * a;
    src[i * 4 + 3] = a;
  }
  stbi_image_free(png);

  rta::ImageView srcView{src.data(), w, h, std::ptrdiff_t(w) * 4};
  const rta::Segmentation seg = rta::segment(srcView, p);

  rta::Segmentation charSeg;
  if (sp.needsPerCharacter()) {
    rta::DetectParams cp = p;
    cp.mode = rta::GroupMode::Character;
    charSeg = rta::segment(srcView, cp);
  }
  const rta::Segmentation& unitSeg = sp.needsPerCharacter() ? charSeg : seg;
  const std::vector<int> unitToGroup = rta::mapUnitsToGroups(unitSeg, seg);
  const std::vector<float> offsets = rta::spacingOffsets(unitSeg, sp);

  std::printf("%dx%d groups=%zu\n", w, h, seg.groups.size());
  if (seg.empty()) return 1;

  void* queue = rta::metalCreateQueue();
  if (!queue) {
    std::printf("no Metal device\n");
    return 1;
  }

  // Source and destination laid out differently from each other, as two host
  // images can be: the kernels must not assume they share a stride.
  HostLayout srcL{w, h, std::ptrdiff_t(w) * 4 + 8, 12};
  HostLayout dstL{w, h, std::ptrdiff_t(w) * 4 + 36, 100};
  std::vector<float> srcHost(srcL.floats(), -7.0f);  // junk outside the image must never be read
  for (int y = 0; y < h; ++y)
    std::memcpy(&srcHost[srcL.index(0, y)], &src[size_t(y) * w * 4], size_t(w) * 4 * sizeof(float));
  std::vector<float> dstInit(dstL.floats(), -3.0f);
  void* srcBuf = rta::metalCreateBuffer(queue, srcHost.size() * sizeof(float), srcHost.data());
  void* dstBuf = rta::metalCreateBuffer(queue, dstInit.size() * sizeof(float), dstInit.data());
  if (!srcBuf || !dstBuf) {
    std::printf("buffer allocation failed\n");
    return 1;
  }
  const rta::MetalImage srcM = srcL.view(srcBuf);
  const rta::MetalImage dstM = dstL.view(dstBuf);

  int failures = 0;

  // ---- fingerprint: stable across calls, and changed by a changed title
  {
    bool ok1 = false, ok2 = false;
    const uint64_t h1 = rta::metalHashAlpha(srcM, 0.5f, queue, &ok1);
    const uint64_t h2 = rta::metalHashAlpha(srcM, 0.5f, queue, &ok2);
    // The same picture moved one sampled pixel: same histogram, different place.
    rta::MetalImage shifted = srcM;
    shifted.offset += 8 * 4;
    shifted.width -= 8;
    bool ok3 = false;
    const uint64_t h3 = rta::metalHashAlpha(shifted, 0.5f, queue, &ok3);
    const bool good = ok1 && ok2 && ok3 && h1 == h2 && h1 != h3;
    std::printf("hash: %016llx %016llx shifted %016llx  %s\n", (unsigned long long)h1,
                (unsigned long long)h2, (unsigned long long)h3, good ? "ok" : "FAILED");
    if (!good) ++failures;
  }

  // ---- readback: the bytes segmentation would be run on
  {
    std::vector<float> back;
    bool good = rta::metalDownload(back, srcM, queue);
    if (good) {
      // Same arithmetic plugin.cpp does on the result.
      const std::ptrdiff_t absStride = srcL.stride;
      const rta::ImageView v{back.data() + absStride * (h - 1), w, h, -absStride};
      for (int y = 0; y < h && good; ++y)
        good = std::memcmp(v.row(y), &src[size_t(y) * w * 4], size_t(w) * 4 * sizeof(float)) == 0;
    }
    std::printf("download: %s\n", good ? "ok" : "FAILED");
    if (!good) ++failures;
  }

  rta::AnimParams anim;  // plugin defaults: dur 12, stagger 2
  anim.in.slideDistance = slide;
  anim.in.startScale = startScale;
  anim.in.startRotation = startRotation;
  anim.in.startBlur = startBlur;
  anim.motionBlur = motionBlur;

  const int owner = 0;
  const auto unitShared = std::make_shared<const rta::Segmentation>(unitSeg);

  const std::vector<int> rank = rta::revealOrder(seg.groups, seg.lineCount, anim.in);
  const std::vector<double> delays = rta::revealDelays(seg.groups, rank, anim.in);
  const int taps = rta::tapCount(anim);
  const rta::RectI window{0, 0, w, h};

  std::vector<float> cpu(size_t(w) * h * 4);
  rta::ImageView cpuView{cpu.data(), w, h, std::ptrdiff_t(w) * 4};

  double cpuMs = 0.0, gpuMs = 0.0;
  using clock = std::chrono::steady_clock;

  for (int f = 0; f < frameCount; ++f) {
    const double t = f;
    std::vector<rta::GroupTransform> xf(seg.groups.size() * size_t(taps));
    for (size_t g = 0; g < seg.groups.size(); ++g)
      rta::transformTaps(rta::Stage::In, delays[g], t, anim.startTime, anim, anim.in,
                         &xf[g * size_t(taps)]);

    std::vector<rta::GroupTransform> uxf;
    std::vector<float> pivots;
    const float* pivotData = nullptr;
    if (sp.any()) {
      rta::applySpacing(unitSeg, seg, unitToGroup, offsets, xf, taps, &uxf, &pivots);
      pivotData = pivots.data();
    }
    const std::vector<rta::GroupTransform>& ct = sp.any() ? uxf : xf;

    auto c0 = clock::now();
    rta::compositeGroups(cpuView, srcView, unitSeg, ct, taps, window, pivotData);
    auto c1 = clock::now();

    {
      rta::MetalLease lease(queue);
      const auto stencil = rta::metalStencil(&owner, unitShared, queue);
      if (!lease.valid() || !stencil ||
          !rta::metalCompositeGroups(dstM, srcM, *stencil, unitSeg.groups, ct, taps, window,
                                     lease, queue, pivotData)) {
        std::printf("frame %2d: metalCompositeGroups FAILED\n", f);
        ++failures;
        continue;
      }
    }
    rta::metalFinish(queue);
    auto c2 = clock::now();
    cpuMs += std::chrono::duration<double, std::milli>(c1 - c0).count();
    gpuMs += std::chrono::duration<double, std::milli>(c2 - c1).count();

    const float* gpu = static_cast<const float*>(rta::metalBufferContents(dstBuf));
    double maxDiff = 0.0, cpuSum = 0.0, gpuSum = 0.0;
    for (int y = 0; y < h; ++y) {
      const float* g = gpu + dstL.index(0, y);
      const float* c = &cpu[size_t(y) * w * 4];
      for (int i = 0; i < w * 4; ++i) {
        maxDiff = std::max(maxDiff, double(std::fabs(c[i] - g[i])));
        cpuSum += c[i];
        gpuSum += g[i];
      }
    }
    // Nothing outside the image may have been written: the padding still holds
    // what it was filled with.
    bool padIntact = true;
    for (size_t i = 0; i < size_t(dstL.lead) && padIntact; ++i) padIntact = gpu[i] == -3.0f;
    for (int y = 0; y < h && padIntact; ++y) {
      const size_t rowEnd = dstL.index(0, y) + size_t(w) * 4;
      for (size_t i = rowEnd; i < rowEnd + size_t(dstL.stride - std::ptrdiff_t(w) * 4); ++i)
        if (gpu[i] != -3.0f) {
          padIntact = false;
          break;
        }
    }
    const bool bad = maxDiff > 0.02 || !std::isfinite(maxDiff) || !padIntact;
    if (bad) ++failures;
    if (bad || f < 6 || f % 6 == 0)
      std::printf("frame %2d: maxDiff=%.6f  cpuSum=%.1f gpuSum=%.1f%s  %s\n", f, maxDiff, cpuSum,
                  gpuSum, padIntact ? "" : "  WROTE OUTSIDE THE IMAGE", bad ? "MISMATCH" : "ok");
  }

  const rta::DeviceMemoryUse mem = rta::deviceMemoryUse();
  std::printf("\nper frame: cpu %.2f ms, metal %.2f ms (incl. wait)   held: stencils %zu KB, "
              "tiles %zu KB in %d\n",
              cpuMs / std::max(1, frameCount), gpuMs / std::max(1, frameCount),
              mem.stencils >> 10, mem.scratch >> 10, mem.scratchCount);
  std::printf("%s\n", failures ? "PARITY FAILURES" : "cpu and metal agree");
  return failures ? 1 : 0;
}
