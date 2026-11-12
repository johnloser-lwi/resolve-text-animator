// GPU render path.
//
// Resolve's pipeline is GPU-resident, so a CPU-only OFX plugin forces a full
// frame down to system memory and back every frame. That readback, not the
// arithmetic, is what stops playback being realtime. Compositing on the GPU
// keeps the frame in VRAM.
//
// Segmentation deliberately stays on the CPU: connected-component labeling with
// union-find is a poor fit for GPUs, and it only runs on a cache miss, so it
// costs one readback when the title changes rather than one per frame.
//
// This header is plain C++ and contains no CUDA types, so plugin.cpp compiles
// with the host compiler.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "animator.h"
#include "image_view.h"
#include "segmentation.h"

namespace rta {

#if RTA_WITH_CUDA

// True if a usable CUDA device is present. Cached after the first call.
bool cudaAvailable();

// Device-resident copy of the label stencil, kept alive across frames so the
// per-frame cost is just the composite kernels.
class CudaSegmentation {
 public:
  CudaSegmentation() = default;
  ~CudaSegmentation();
  CudaSegmentation(const CudaSegmentation&) = delete;
  CudaSegmentation& operator=(const CudaSegmentation&) = delete;

  // Uploads the label image and label->group table. Safe to call repeatedly;
  // reallocates only when the size changes.
  bool upload(const Segmentation& seg);
  void reset();
  bool valid() const { return labels_ != nullptr; }

  const int32_t* labels() const { return labels_; }
  const int32_t* labelToGroup() const { return labelToGroup_; }
  int width() const { return width_; }
  int height() const { return height_; }
  size_t bytes() const {
    return (size_t(width_) * size_t(height_) + size_t(labelCount_)) * sizeof(int32_t);
  }

 private:
  int32_t* labels_ = nullptr;
  int32_t* labelToGroup_ = nullptr;
  int width_ = 0, height_ = 0;
  int labelCount_ = 0;
};

// Pulls a device image into a host buffer laid out exactly as the device one,
// so the caller can wrap it in an ImageView (including a negative stride to
// flip OFX's bottom-up rows). Only used on a segmentation cache miss.
bool cudaDownload(std::vector<float>& out, const float* srcDev,
                  std::ptrdiff_t srcStrideFloats, int width, int height);

// Persistent device scratch. cudaMalloc synchronises and costs tens of
// microseconds, so the per-frame hash must not allocate.
class CudaScratch {
 public:
  CudaScratch() = default;
  ~CudaScratch();
  CudaScratch(const CudaScratch&) = delete;
  CudaScratch& operator=(const CudaScratch&) = delete;

  // Lazily allocated 8-byte accumulator; null if allocation failed.
  unsigned long long* hashAccum();

  // Grow-only device buffer for per-frame transform uploads.
  void* tapBuffer(size_t bytes);

  // Two grow-only tiles for the defocus. A separable blur reads one and writes
  // the other, so it needs somewhere to land that is not what it is reading.
  void* tileBuffer(int which, size_t bytes);

 private:
  unsigned long long* accum_ = nullptr;
  void* taps_ = nullptr;
  size_t tapBytes_ = 0;
  void* tiles_[2] = {nullptr, nullptr};
  size_t tileBytes_[2] = {0, 0};
  bool tried_ = false;
};

// ------------------------------------------------- device memory, shared
// Both of the above used to belong to an effect instance and lived as long as
// it did. The host keeps an instance for every use of the effect, so a long
// timeline held a stencil and two blur tiles for every title in it, all at
// once, and nothing was given back. Now they belong to a per-device store, sized
// by what renders at once rather than by how many instances exist.

// A scratch for one render. The store holds one per render that has ever run
// at the same time as others, so blur tiles are as many as concurrent renders.
//
// Ending a lease does not wait for the GPU. It records an event on the stream
// instead, and the next lease of the same scratch makes its own stream wait on
// it -- so the kernels still reading the tiles finish first, without stalling
// the CPU here.
class ScratchLease {
 public:
  explicit ScratchLease(void* stream);
  ~ScratchLease();
  ScratchLease(const ScratchLease&) = delete;
  ScratchLease& operator=(const ScratchLease&) = delete;

  // Null only if the store could not make one.
  CudaScratch* get() const;

  struct Slot;  // a scratch and the event that says when its last user finished

 private:
  Slot* slot_ = nullptr;
  int device_ = 0;
  void* stream_ = nullptr;
};

// The device stencil of `seg` for `owner`, uploaded only when what is held is
// not already that segmentation. Over the budget, the stencils of owners that
// have not rendered for a while are freed; they upload again if they return.
// Shared, so an eviction by another render cannot free one mid-frame. Null if
// the upload failed.
std::shared_ptr<const CudaSegmentation> deviceStencil(const void* owner,
                                                      const std::shared_ptr<const Segmentation>& seg);

// The owner is gone. Its stencil is freed by the next render on its device --
// not here, because the host may destroy an instance on a thread where its
// CUDA context is not current, and a free there goes to the wrong context.
void forgetStencil(const void* owner);

// Content fingerprint of the alpha channel, computed on-device so detecting a
// changed title costs an 8-byte readback rather than a whole frame. Combining
// is commutative (atomic sum of per-pixel mixes) because kernel order is not
// deterministic; that is fine for cache invalidation.
uint64_t cudaHashAlpha(const float* srcDev, std::ptrdiff_t srcStrideFloats, int width, int height,
                       float threshold, CudaScratch& scratch, void* stream, bool* ok);

bool cudaClearWindow(float* dstDev, std::ptrdiff_t dstStrideFloats, int width, int height,
                     const RectI& window, void* stream);

// Clears `window` then composites every visible group. Mirrors
// compositeGroups() exactly, including the group-major `taps` layout.
// Strides are in floats and may be negative. Returns false if any CUDA call
// failed, in which case nothing is guaranteed about dst.
bool cudaCompositeGroups(float* dstDev, std::ptrdiff_t dstStrideFloats, const float* srcDev,
                         std::ptrdiff_t srcStrideFloats, int width, int height,
                         const CudaSegmentation& devSeg, const std::vector<Group>& groups,
                         const std::vector<GroupTransform>& transforms, int taps,
                         const RectI& window, CudaScratch& scratch, void* stream,
                         const float* pivots = nullptr);

// Dims the source into dst and draws a coloured box per group, matching the
// CPU diagnostics overlay.
bool cudaDrawDiagnostics(float* dstDev, std::ptrdiff_t dstStrideFloats, const float* srcDev,
                         std::ptrdiff_t srcStrideFloats, int width, int height,
                         const std::vector<Group>& groups, const RectI& window, int lineWidth,
                         float slantTan, int yMid, void* stream);

#endif  // RTA_WITH_CUDA

}  // namespace rta
