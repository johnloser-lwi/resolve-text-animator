// Metal render path -- the Mac counterpart of cuda_compositor.h.
//
// Same reasoning as the CUDA path: Resolve's pipeline is GPU-resident, so a
// CPU-only plugin forces every frame down to system memory and back, and that
// round trip is what stops playback being realtime. Compositing on the GPU keeps
// the frame where it is.
//
// The split is the same too. Segmentation stays on the CPU and runs only on a
// cache miss; per frame the GPU fingerprints the alpha (an 8-byte readback) and
// composites. The kernels are the CPU compositor's arithmetic written once more
// in Metal Shading Language -- see compositor.cpp, which is the reference that
// tools/metaltest checks them against.
//
// One difference from CUDA shapes the whole interface: an OFX host hands Metal
// images over as an id<MTLBuffer>, not a pointer. There is no address to do
// arithmetic on, so a view here is a buffer plus the float offset of its pixel
// (0,0), with the same signed row stride an ImageView has.
//
// This header is plain C++ with no Objective-C or Metal types, so plugin.cpp
// compiles with the ordinary C++ compiler. Handles are passed as void*:
// `queue` is an id<MTLCommandQueue> and `buffer` an id<MTLBuffer>, both
// borrowed from the host and never retained beyond the call.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "animator.h"
#include "device_memory.h"
#include "image_view.h"
#include "segmentation.h"

namespace rta {

#if RTA_WITH_METAL

// A region of a host Metal buffer, laid out like an ImageView: pixel (x, y) of
// the region starts at float index  offset + rowStride * y + 4 * x.
struct MetalImage {
  void* buffer = nullptr;        // id<MTLBuffer>, borrowed
  int64_t offset = 0;            // floats from the buffer's start to pixel (0,0)
  std::ptrdiff_t rowStride = 0;  // floats; negative for bottom-up rows
  int width = 0, height = 0;

  bool valid() const { return buffer != nullptr && width > 0 && height > 0; }
};

// True if this machine has a Metal device at all. Cached after the first call.
// Says nothing about whether the kernels compile; that is learned on the first
// render, and a failure there renders nothing rather than garbage.
bool metalAvailable();

// The device copy of a segmentation's label stencil. Opaque: only the .mm file
// knows it holds MTLBuffers.
class MetalStencil;

// Blur tiles for one render, borrowed from a shared per-device store and handed
// back when the GPU has finished with them -- not when this object dies, which
// is usually long before. Mirrors ScratchLease in cuda_compositor.h.
class MetalLease {
 public:
  explicit MetalLease(void* queue);
  ~MetalLease();
  MetalLease(const MetalLease&) = delete;
  MetalLease& operator=(const MetalLease&) = delete;

  bool valid() const { return slot_ != nullptr; }

  struct Slot;

 private:
  friend bool metalCompositeGroups(const MetalImage&, const MetalImage&, const MetalStencil&,
                                   const std::vector<Group>&, const std::vector<GroupTransform>&,
                                   int, const RectI&, MetalLease&, void*, const float*);
  Slot* slot_ = nullptr;
  uint64_t device_ = 0;
  // Set once a command buffer that uses the tiles is committed; the slot then
  // goes back to the store from that buffer's completion handler instead.
  bool handedOff_ = false;
};

// The device stencil of `seg` for `owner`, uploaded only when what is held is
// not already that segmentation. Shares the CUDA store's policy: over budget,
// stencils of owners that have not rendered for a while are dropped. Null if
// the upload failed.
std::shared_ptr<const MetalStencil> metalStencil(const void* owner,
                                                 const std::shared_ptr<const Segmentation>& seg,
                                                 void* queue);

// The owner is gone; its stencil is dropped on the next render.
void metalForgetStencil(const void* owner);

// Content fingerprint of the alpha channel, computed on the GPU. Waits for the
// queue to reach it -- the one per-frame wait in this path, as on CUDA -- and
// reads back 8 bytes.
uint64_t metalHashAlpha(const MetalImage& src, float threshold, void* queue, bool* ok);

// Pulls a region into host memory laid out exactly as the device one, so the
// caller can wrap it in an ImageView: `out` starts at the region's lowest-
// addressed row, |rowStride| floats per row. Only used on a segmentation miss.
bool metalDownload(std::vector<float>& out, const MetalImage& src, void* queue);

bool metalClearWindow(const MetalImage& dst, const RectI& window, void* queue);

// Clears `window` then composites every visible group. Mirrors compositeGroups()
// exactly, including the group-major `taps` layout. Commits its work to `queue`
// and returns without waiting for it. False if anything failed to encode, in
// which case nothing is guaranteed about dst.
bool metalCompositeGroups(const MetalImage& dst, const MetalImage& src,
                          const MetalStencil& stencil, const std::vector<Group>& groups,
                          const std::vector<GroupTransform>& transforms, int taps,
                          const RectI& window, MetalLease& lease, void* queue,
                          const float* pivots = nullptr);

// Blocks until everything committed to `queue` so far has run. For the test
// harness, which reads the result back on the CPU; the plugin never calls it.
void metalFinish(void* queue);

// A command queue on the default device, for the harness. Retained; the
// harness lives for one run and never frees it.
void* metalCreateQueue();

// Shared-storage buffer for the harness, initialised from `data` if given.
void* metalCreateBuffer(void* queue, size_t bytes, const void* data);
const void* metalBufferContents(void* buffer);

#endif  // RTA_WITH_METAL

}  // namespace rta
