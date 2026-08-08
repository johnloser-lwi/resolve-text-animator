// CPU vs CUDA compositor parity.
//
// The CPU path is the one segtest exercises, so it is the reference. The CUDA
// kernels are a separate implementation of the same maths and have never been
// checked against it -- and in Resolve every frame goes through them. Early
// frames of a reveal are exactly where they would diverge: large slide offsets,
// low opacity, groups that have not started yet.
//
//   cudatest <in.png> [--mode char|word|line] [--frames N]
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "animator.h"
#include "compositor.h"
#include "cuda_compositor.h"
#include "segmentation.h"

#include <cuda_runtime.h>

namespace {

bool uploadImage(const std::vector<float>& host, int w, int h, float** dev, size_t* pitch) {
  if (cudaMallocPitch(dev, pitch, size_t(w) * 4 * sizeof(float), size_t(h)) != cudaSuccess)
    return false;
  return cudaMemcpy2D(*dev, *pitch, host.data(), size_t(w) * 4 * sizeof(float),
                      size_t(w) * 4 * sizeof(float), size_t(h),
                      cudaMemcpyHostToDevice) == cudaSuccess;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: cudatest <in.png> [--mode char|word|line] [--frames N]\n");
    return 2;
  }
  rta::DetectParams p;
  int frameCount = 24;
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--mode") && i + 1 < argc) {
      const char* m = argv[++i];
      p.mode = !std::strcmp(m, "char")   ? rta::GroupMode::Character
               : !std::strcmp(m, "line") ? rta::GroupMode::Line
                                         : rta::GroupMode::Word;
    } else if (!std::strcmp(argv[i], "--frames") && i + 1 < argc) {
      frameCount = std::atoi(argv[++i]);
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
  std::printf("%dx%d groups=%zu\n", w, h, seg.groups.size());
  if (seg.empty()) return 1;

  rta::AnimParams anim;  // plugin defaults: dur 12, stagger 2
  anim.in.slideDistance = 45.0;

  rta::CudaSegmentation devSeg;
  rta::CudaScratch scratch;
  if (!devSeg.upload(seg)) {
    std::fprintf(stderr, "devSeg upload failed\n");
    return 1;
  }

  float *dSrc = nullptr, *dDst = nullptr;
  size_t srcPitch = 0, dstPitch = 0;
  if (!uploadImage(src, w, h, &dSrc, &srcPitch) ||
      cudaMallocPitch(&dDst, &dstPitch, size_t(w) * 4 * sizeof(float), size_t(h)) != cudaSuccess) {
    std::fprintf(stderr, "device alloc failed\n");
    return 1;
  }

  const std::vector<int> rank = rta::revealOrder(seg.groups, seg.lineCount, anim.in);
  const int taps = rta::tapCount(anim);
  const rta::RectI window{0, 0, w, h};

  std::vector<float> cpu(size_t(w) * h * 4), gpu(size_t(w) * h * 4);
  rta::ImageView cpuView{cpu.data(), w, h, std::ptrdiff_t(w) * 4};

  int failures = 0;
  for (int f = 0; f < frameCount; ++f) {
    const double t = f;
    std::vector<rta::GroupTransform> xf(seg.groups.size() * size_t(taps));
    for (size_t g = 0; g < seg.groups.size(); ++g)
      rta::transformTaps(rta::Stage::In, rank[g], t, anim.startTime, anim, anim.in,
                         &xf[g * size_t(taps)]);

    rta::compositeGroups(cpuView, srcView, seg, xf, taps, window);

    const std::ptrdiff_t srcStrideF = std::ptrdiff_t(srcPitch / sizeof(float));
    const std::ptrdiff_t dstStrideF = std::ptrdiff_t(dstPitch / sizeof(float));
    if (!rta::cudaCompositeGroups(dDst, dstStrideF, dSrc, srcStrideF, w, h, devSeg, seg.groups, xf,
                                  taps, window, scratch, nullptr)) {
      std::printf("frame %2d: cudaCompositeGroups FAILED\n", f);
      ++failures;
      continue;
    }
    cudaDeviceSynchronize();
    cudaMemcpy2D(gpu.data(), size_t(w) * 4 * sizeof(float), dDst, dstPitch,
                 size_t(w) * 4 * sizeof(float), size_t(h), cudaMemcpyDeviceToHost);

    double maxDiff = 0.0, cpuSum = 0.0, gpuSum = 0.0;
    for (size_t i = 0; i < cpu.size(); ++i) {
      maxDiff = std::max(maxDiff, double(std::fabs(cpu[i] - gpu[i])));
      cpuSum += cpu[i];
      gpuSum += gpu[i];
    }
    const bool bad = maxDiff > 0.02;
    if (bad) ++failures;
    if (bad || f < 6 || f % 6 == 0)
      std::printf("frame %2d: maxDiff=%.4f  cpuSum=%.1f gpuSum=%.1f  %s\n", f, maxDiff, cpuSum,
                  gpuSum, bad ? "MISMATCH" : "ok");
  }

  std::printf("\n%s\n", failures ? "PARITY FAILURES" : "cpu and cuda agree");
  return failures ? 1 : 0;
}
