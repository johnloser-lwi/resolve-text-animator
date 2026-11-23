// What the GPU store holds, for the viewer's memory readout. Shared by the CUDA
// and Metal paths -- only one is ever built into a plugin -- so the overlay
// reads the same figures whichever it is.
#pragma once

#include <cstddef>

namespace rta {

// What the store holds right now, across every device and every instance.
// Shown in the viewer so the plugin's share of the GPU can be told apart from
// the host's own, which the OS reports as one number.
struct DeviceMemoryUse {
  size_t stencils = 0;
  size_t scratch = 0;  // blur tiles and per-render buffers
  int scratchCount = 0;
};

#if RTA_WITH_CUDA || RTA_WITH_METAL
DeviceMemoryUse deviceMemoryUse();
#endif

}  // namespace rta
