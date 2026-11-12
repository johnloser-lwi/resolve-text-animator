// Exercises the shared device store: the eviction policy on its own, then the
// store itself, counting live stencils exactly through weak references (VRAM
// totals are shared with every other program on the GPU, so they are noisy),
// then tile trimming and idle release through the store's own readout.
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_compositor.h"
#include "lru_budget.h"

static int fails = 0;
static void check(bool c, const char* msg) {
  std::printf("  %s %s\n", c ? "ok:  " : "FAIL:", msg);
  if (!c) ++fails;
}
static long mb(size_t b) { return long((b + (size_t(1) << 19)) >> 20); }

int main() {
  std::printf("LruBudget\n");
  {
    int A, B, C;
    rta::LruBudget b;
    b.touch(&A, 100, 0.0);
    b.touch(&B, 100, 1.0);
    b.touch(&C, 100, 10.0);
    auto v = b.victims(150, 10.0, 3.0, &C);
    check(v.size() == 2 && v[0] == &A && v[1] == &B, "over budget: oldest first until it fits");
    v = b.victims(250, 10.0, 3.0, &C);
    check(v.size() == 1 && v[0] == &A, "drops only as many as needed");
    v = b.victims(150, 10.0, 20.0, nullptr);
    check(v.empty(), "recent use is never dropped, even over budget");
    v = b.victims(1000, 10.0, 3.0, nullptr);
    check(v.empty(), "under budget drops nothing");
    v = b.victims(0, 10.0, 0.0, &A);
    check(v.size() == 2 && v[0] != &A && v[1] != &A, "the caller is never its own victim");
  }

  std::printf("Stencils (1080p, ~8 MB each, budget 256 MB)\n");
  cudaFree(nullptr);
  rta::Segmentation s;
  s.width = 1920;
  s.height = 1080;
  s.labelImage.assign(size_t(1920) * 1080, 0);
  s.labelToGroup = {0};
  auto fresh = [&] { return std::make_shared<const rta::Segmentation>(s); };

  const int N = 100;
  std::vector<int> owners(N + 2);
  std::vector<std::shared_ptr<const rta::Segmentation>> segs;
  std::vector<std::weak_ptr<const rta::CudaSegmentation>> live;
  auto alive = [&] {
    int n = 0;
    for (auto& w : live) n += !w.expired();
    return n;
  };

  bool allOk = true;
  for (int i = 0; i < N; ++i) {
    segs.push_back(fresh());
    rta::ScratchLease lease(nullptr);
    auto st = rta::deviceStencil(&owners[i], segs[i]);
    allOk = allOk && lease.get() && st;
    live.push_back(st);
  }
  check(allOk, "every render got a scratch and a stencil");
  std::printf("  alive after %d quick renders: %d; readout says %ld MB\n", N, alive(),
              mb(rta::deviceMemoryUse().stencils));
  check(alive() == N, "all used within the recent window, so none evicted");

  {
    auto a = rta::deviceStencil(&owners[N - 1], segs[N - 1]);
    auto b = rta::deviceStencil(&owners[N - 1], segs[N - 1]);
    check(a && a == b, "same segmentation again: same stencil, no re-upload");
    auto c = rta::deviceStencil(&owners[N - 1], fresh());
    check(c && c != a, "a new segmentation for the same owner is uploaded");
  }

  std::printf("  waiting out the recent window...\n");
  std::this_thread::sleep_for(std::chrono::milliseconds(3500));
  live.push_back(rta::deviceStencil(&owners[N], fresh()));
  const int fits = int((size_t(256) << 20) / (size_t(1920) * 1080 * 4 + 4));
  std::printf("  alive after one render past the window: %d (budget fits %d); readout %ld MB\n",
              alive(), fits, mb(rta::deviceMemoryUse().stencils));
  check(alive() <= fits && alive() >= fits - 2, "idle stencils evicted down to the budget");
  check(mb(rta::deviceMemoryUse().stencils) <= 256, "readout agrees: stencils within budget");

  {
    auto held = rta::deviceStencil(&owners[N], fresh());
    for (int i = 0; i <= N; ++i) rta::forgetStencil(&owners[i]);
    live.push_back(rta::deviceStencil(&owners[N + 1], fresh()));
    check(held && held->valid(), "a stencil a frame holds stays valid after its owner is forgotten");
  }
  std::printf("  alive after forgetting every other owner: %d\n", alive());
  check(alive() == 1, "forgotten owners freed at the next render");

  std::printf("Scratch leases\n");
  {
    rta::CudaScratch* first = nullptr;
    {
      rta::ScratchLease a(nullptr);
      first = a.get();
    }
    rta::ScratchLease b(nullptr);
    check(b.get() == first, "sequential renders reuse one scratch");
    rta::ScratchLease c(nullptr);
    check(c.get() && c.get() != b.get(), "concurrent renders get their own");
  }

  std::printf("Tile trimming\n");
  const size_t big = size_t(200) << 20, small = size_t(10) << 20;
  {
    rta::ScratchLease a(nullptr);
    check(a.get()->tileBuffer(0, big, nullptr) && a.get()->tileBuffer(1, big, nullptr), "two 200 MB tiles");
  }
  long held = mb(rta::deviceMemoryUse().scratch);
  std::printf("  after a heavy-blur frame: blur buffers %ld MB\n", held);
  check(held >= 480, "a frame that used big tiles keeps them for the next");
  {
    rta::ScratchLease a(nullptr);
    a.get()->tileBuffer(0, big, nullptr);
  }
  held = mb(rta::deviceMemoryUse().scratch);
  std::printf("  after a frame that used one of them: %ld MB\n", held);
  check(held >= 240 && held < 480, "the tile it did not use is freed");
  {
    rta::ScratchLease a(nullptr);
    a.get()->tileBuffer(0, small, nullptr);
  }
  held = mb(rta::deviceMemoryUse().scratch);
  std::printf("  after a light frame: %ld MB\n", held);
  check(held < 64, "a tile far bigger than the frame needed is freed");

  std::printf("Idle scratches\n");
  {
    rta::ScratchLease a(nullptr), b(nullptr);
    a.get()->tileBuffer(0, size_t(100) << 20, nullptr);
    b.get()->tileBuffer(0, size_t(100) << 20, nullptr);
  }
  const rta::DeviceMemoryUse two = rta::deviceMemoryUse();
  std::printf("  two overlapping renders: %d scratches, %ld MB\n", two.scratchCount, mb(two.scratch));
  std::printf("  waiting out the idle time...\n");
  std::this_thread::sleep_for(std::chrono::milliseconds(10500));
  { rta::ScratchLease a(nullptr); }
  const rta::DeviceMemoryUse after = rta::deviceMemoryUse();
  std::printf("  one render later: %ld MB\n", mb(after.scratch));
  check(mb(after.scratch) <= mb(two.scratch) - 100, "the idle scratch gave its tiles back");

  std::printf("\n%s\n", fails ? "STORE TEST FAILURES" : "store behaves");
  return fails ? 1 : 0;
}
