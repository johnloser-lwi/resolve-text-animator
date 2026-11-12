# Memory that grows over a long render: diagnosis and fix

Written from Text Animator (commits `b815a58` and the tile-trimming follow-up) as a handoff for other OFX plugins
that may have the same problem. Everything here is about *retention*, not a
classic leak: nothing was lost track of, it was just never given back.

## The symptom

- RAM and VRAM climb steadily while working in a project that uses the effect a
  lot, and are not released.
- A long render (30 minutes of timeline) gets slower towards the end.
- It is worse in projects where an expensive option is used heavily (here: blur,
  which brings in extra and larger GPU buffers).

## The cause

**The host keeps an effect instance alive for every use of the effect, for the
whole session.** Resolve does not destroy the instance for a clip once it has
rendered it. So anything an instance caches lives as long as the project is open.

Text Animator cached per instance, by design (to avoid redoing work every frame):

| Held per instance | Where | Size |
|---|---|---|
| Label image per grouping mode (up to 3) | RAM | 8 MB at 1080p, 33 MB at UHD, each |
| Device copy of the label image | VRAM | same |
| Two blur tiles, **grow-only**, allocated at **2x** the largest tile drawn | VRAM | up to 2x a full frame, each |

Each of these was reasonable for one instance. Multiplied by every title in a
long timeline, it is gigabytes of RAM and VRAM held for titles that finished
rendering long ago. As VRAM fills, the driver and the host start evicting and
paging, so later frames render slower.

The general rule: **per-instance caches are bounded by the number of instances
the host keeps, which is unbounded.** The only thing that should bound them is
what renders *at the same time*.

## How to check another plugin

Look for state owned by the effect-instance class (or by a per-instance cache
struct) that is not released until the destructor:

1. **GPU allocations owned by the instance.** Grep for `cudaMalloc`,
   `MTLBuffer`/`newBufferWithLength`, `clCreateBuffer`, `glGenTextures`, and see
   who owns the pointer. Owned by the instance and freed only in its destructor =
   this problem.
2. **Grow-only buffers.** A buffer that reallocates only when a bigger request
   arrives, especially one that rounds up (`bytes * 2`). It holds the largest
   size it has ever seen, forever.
3. **Per-instance analysis caches in RAM**: images, label maps, meshes,
   simulation state, anything per-pixel. Check the size at UHD.
4. **Which options make it worse.** Blur, motion blur, high sample counts and
   large padding typically add buffers or widen them.
5. **Destructors that free GPU memory.** The host may destroy an instance on a
   thread where its GPU context is not current, so the free can go to the wrong
   context and quietly leak. This is a separate leak with the same symptom.

A quick estimate: (bytes per instance) x (number of uses of the effect in the
project). If that is in the gigabytes, this is the problem.

## The fix (three parts)

### 1. Lease per-render scratch from a shared pool

Buffers needed only *during* a render (blur tiles, temporary accumulators,
upload staging) do not belong to an instance. Keep a pool per GPU device; a
render takes one scratch when it starts and returns it when it ends. The pool
grows to the number of renders that ever ran *at once*, not the number of
instances.

**The trap: GPU work is asynchronous.** When a render returns, its kernels may
still be reading the scratch on the GPU. If the next render (on another stream)
takes the same scratch immediately, it overwrites tiles still in use. Two ways
to handle it:

- *Cheap and correct:* when a lease ends, `cudaEventRecord(event, stream)`. When
  the next lease takes that scratch, `cudaStreamWaitEvent(newStream, event)`.
  The GPU orders the work; the CPU never waits.
- *Simple fallback:* `cudaStreamSynchronize(stream)` before returning the
  scratch. Correct, but stalls the render thread until the GPU finishes.

Also: round grow-only buffers up by a fraction (we use 1.25x), not 2x. A tile
can be a whole frame; doubling that holds two frames' worth for good.

**Pooling bounds how many buffers exist, not how big each one grows.** This was
the second round of the fix: with only the pool, VRAM still crept up late in a
render, because every pooled tile kept the largest size it had ever drawn. Two
more rules close that:

- *Trim at the end of each lease.* Track the largest size asked of **each**
  buffer during the lease (per buffer, not one peak for all -- a blur-only
  buffer must be freeable on a frame without blur while the other stays busy).
  Free any buffer over 64 MB that is more than 4x that peak. Four, so a blur
  that shrinks frame by frame reallocates a few times on the way down, not
  every frame.
- *Release idle pool entries.* Extra entries exist only because renders once
  overlapped. When one has been idle for 10 s, free its buffers, at the next
  lease on that device.

Both free with `cudaFree`, which waits for the device, so nothing still queued
can be reading what is freed.

### 2. Keep reusable per-instance data under one global budget (LRU)

Data worth keeping between frames of the *same* clip (analysis results, device
copies of them) goes in a store with a byte budget shared by all instances:

- On each use, record (owner, bytes, time).
- When the total is over budget, drop the **longest-unused first**...
- ...but **never anything used in the last few seconds** (we use 3 s). Several
  titles on screen together are all in use every frame. Without this window they
  would evict each other every frame and re-analyse constantly (thrashing). With
  it, the budget is allowed to overflow instead, which is the right failure.
- A dropped owner simply recomputes once when it renders again.

Budgets used here: 1 GB for the host analyses, 256 MB for the device stencils.
The device budget can be small when the host copy is kept separately: a dropped
device copy costs one upload (about a millisecond at 1080p) to bring back, while
a dropped host analysis costs a full re-detection.

### 3. Make the cross-thread parts safe

**The rule that matters most: never make a GPU call while holding a lock that
any other thread can wait on.** The first version of this fix broke it and hung
Resolve on the way into the Fusion page. The cycle:

1. A render thread holds the store's lock and calls `cudaFree` (or a plain
   `cudaMemcpy`, or `cudaMalloc` in some cases). Those wait for the **whole
   device** to go idle -- the host's own GPU work included.
2. The host's GPU work can be waiting on one of the host's threads (a host
   function queued on a stream, interop with the viewer). Resolve does this.
3. That thread -- here the UI thread, drawing the viewer overlay, which read the
   memory readout -- is waiting for the store's lock. Nobody can move.

So:

- The shared lock guards **bookkeeping only**. Decide under the lock, then let
  go, then upload, allocate or free. Collect evicted objects into a local list
  declared *before* the lock, so their destructors (which free) run after it is
  released.
- Anything the UI thread reads (the readout) is an **atomic**, never behind a
  lock a render can hold.
- In render paths, free with **stream-ordered** calls (`cudaMallocAsync` /
  `cudaFreeAsync` on the render's stream, where
  `cudaDevAttrMemoryPoolsSupported`): the free is queued behind that stream's
  work instead of waiting for the whole device. Where it is unsupported, fall
  back to `cudaFree`, but still never under a lock.
- When freeing something whose last work ran on another stream, make the
  current stream wait on that stream's event first (`cudaStreamWaitEvent`),
  then free on the current stream.

Then the rest:

- **Never free GPU memory from a thread that may not have the host's context.**
  In the destructor, only *mark* the owner as gone. The next render on that
  device (where the host's context is current) frees it.
- **Hold device data by `shared_ptr` and hand a copy to the frame.** Then
  eviction by another render only drops the store's reference; the memory is
  freed when the frame using it lets go.
- **Identify cached data by a `weak_ptr` to its source, not by its address.**
  A freed analysis and a new one can land at the same address; a `weak_ptr` to
  the freed one is expired, so the stale device copy is never reused by mistake.
- **Only `try_lock` another instance's cache when evicting.** If it is locked,
  that instance is rendering, so it is not idle anyway, and skipping it cannot
  deadlock. Never call the sweep while holding your own cache lock.
- **Never destroy the global stores.** Allocate them with `new` and leave them.
  A static destructor runs while the host unloads the plugin, when the CUDA
  runtime may already be gone; the process exit returns the memory anyway.

### 4. Show the plugin's own share

Task Manager reports every plugin's memory as the host's, so a user cannot tell
which plugin is growing. The stores can report what they hold. Text Animator
draws it in the viewer with Show Detection on, top right: RAM held by all
instances' analyses, and GPU memory split into stencils and blur buffers. After
a render it shows what is left. If that stays small while the host keeps
growing, the growth is not this plugin's. Read it from atomics (see 3): the
viewer draws on the UI thread.

## Where the code is

In this repository:

| File | What to take from it |
|---|---|
| `src/lru_budget.h` | The eviction policy. Host-independent, header-only, no locking (callers lock). Copy as is. |
| `src/cuda_compositor.h` / `.cu` (search `device memory, shared`) | `ScratchLease` (pool + event handoff), `deviceStencil` (budgeted device cache keyed by owner and `weak_ptr` to source), `forgetStencil` (deferred free). |
| `src/plugin.cpp` (search `retainAnalysis`) | The host-side budget: a registry of instance caches, `retainAnalysis` after each render, `releaseAnalysis` in the destructor. |
| `tools/cudatest.cpp` | The parity test, now drawing through the store, as the plugin does. |
| `tools/storetest.cpp` | The store's behaviour, counted exactly: budgets, trimming, idle release, the readout. |
| `tools/deadlocktest.cpp` | Reproduces the hang without a host: GPU work that waits on a "UI thread" which reads the readout, while render threads hammer the store. Copy this first -- it is the cheapest way to find out whether another plugin can hang its host. |

## How it was verified

- **Output unchanged:** detection for every test image in every mode and an
  animated, blurred strip are byte-identical to before; the CPU/GPU parity test
  (now through the store) gives the same checksum across blur, motion blur,
  per-character and tracking cases.
- **The store's behaviour, measured exactly.** Do not judge this by VRAM totals:
  `cudaMemGetInfo` counts every program on the GPU, and the numbers moved by
  hundreds of MB during the test from other apps alone. Instead the test holds a
  `weak_ptr` to every stencil the store returns and counts the live ones:
  - 100 stencils used within the 3 s window: all 100 alive (by design).
  - One render after the window: trimmed to 63, within the budget of 64.
  - Every other owner forgotten, then one render: 1 alive. A stencil a frame
    still held stayed valid until the frame released it.
  - Sequential leases reuse one scratch; overlapping leases get their own.
- **The hang, reproduced and fixed.** `deadlocktest` over 8 seconds:

  | Store | Frames | Readouts | GPU waits stuck on the UI thread |
  |---|---|---|---|
  | Lock held across GPU calls (the version that hung Resolve) | 4 | 2 | deadlocked |
  | Bookkeeping-only lock, atomic readout | ~540 | 515 | 0 |

In Resolve, the first version visibly freed VRAM during a long render, which
is what showed retention was the cause of the creep; the trimming and the
deadlock fix after it are verified by the tests above and await a long render.

## Applying it elsewhere: order of work

1. Estimate bytes per instance at UHD, with the expensive options on.
2. Move per-render scratch to a leased pool first. This is usually the biggest
   saving (blur tiles here) and the least risky. Trim pooled buffers per lease
   and release idle ones, or the pool keeps its largest-ever sizes.
3. Put reusable per-instance data under a global LRU budget with a recent-use
   window.
4. Move GPU frees out of destructors.
5. Check every lock the new code adds: **no GPU call while it is held**, and
   nothing the UI thread reads behind it. Then run `deadlocktest` against the
   plugin's store. A shared store is new shared state, and shared state is how
   one plugin instance gets to hang the whole host.
6. Verify output is unchanged with the plugin's own parity or regression tests,
   then count live objects through `weak_ptr`s rather than reading VRAM totals.
