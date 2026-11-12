// Reproduces the Fusion-page hang without Resolve.
//
// A host can queue GPU work that cannot finish until one of its own threads
// acts -- here, a host function on a "host stream" that waits for the "UI
// thread" to tick. The UI thread reads the memory readout every tick, as the
// viewer does. Render threads hammer the store. If the store ever holds its
// lock across a call that waits for the device, the UI thread blocks on the
// lock, stops ticking, the host function never returns, the device never goes
// idle, and the render holding the lock never gets out: a cycle.
//
// Host functions give up after 3 s instead of hanging forever, and count it, so
// the test always finishes. Any timeout is a deadlock that would have been real.
// Uses only the store's public API, so it can be pointed at an older store
// (build it against that version's cuda_compositor.cu) to see the hang.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_compositor.h"

using clk = std::chrono::steady_clock;
static std::atomic<long> uiTicks{0};
static std::atomic<int> timeouts{0};
static std::atomic<bool> running{true};

static void CUDART_CB waitForUi(void*) {
  const long seen = uiTicks.load();
  const auto deadline = clk::now() + std::chrono::seconds(3);
  while (uiTicks.load() == seen && running.load()) {
    if (clk::now() > deadline) {
      timeouts.fetch_add(1);
      return;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  }
}

int main() {
  cudaFree(nullptr);
  rta::Segmentation s;
  s.width = 1920;
  s.height = 1080;
  s.labelImage.assign(size_t(1920) * 1080, 0);
  s.labelToGroup = {0};

  // The host's own stream: created blocking, as a default stream is, so the
  // legacy default stream (which a plain cudaMemcpy uses) orders behind it.
  cudaStream_t hostStream;
  cudaStreamCreate(&hostStream);

  std::thread pump([&] {
    while (running) {
      cudaLaunchHostFunc(hostStream, waitForUi, nullptr);
      cudaStreamSynchronize(hostStream);
    }
  });

  std::atomic<long> readouts{0};
  std::thread ui([&] {
    while (running) {
      volatile size_t sink = rta::deviceMemoryUse().stencils;  // what the viewer draws
      (void)sink;
      readouts.fetch_add(1);
      uiTicks.fetch_add(1);
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  });

  std::atomic<long> frames{0};
  std::vector<std::thread> renders;
  std::vector<int> owners(4 * 40);
  for (int t = 0; t < 4; ++t) {
    renders.emplace_back([&, t] {
      cudaStream_t st;
      cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
      int i = 0;
      while (running) {
        // A new title every frame: every frame uploads, and over the budget
        // the store evicts, so both GPU calls it ever makes keep happening.
        rta::ScratchLease lease(st);
        auto seg = std::make_shared<const rta::Segmentation>(s);
        auto dev = rta::deviceStencil(&owners[t * 40 + (i % 40)], seg);
        if (i % 17 == 0) rta::forgetStencil(&owners[t * 40 + ((i + 20) % 40)]);
        ++i;
        frames.fetch_add(1);
      }
      cudaStreamSynchronize(st);
      cudaStreamDestroy(st);
    });
  }

  std::this_thread::sleep_for(std::chrono::seconds(8));
  running = false;
  for (auto& r : renders) r.join();
  ui.join();
  pump.join();
  std::printf("frames %ld, readouts %ld, host-stream waits that timed out: %d\n", frames.load(),
              readouts.load(), timeouts.load());
  std::printf("%s\n", timeouts.load() ? "DEADLOCK (broken only by the 3 s timeout)" : "no deadlock");
  return timeouts.load() ? 1 : 0;
}
