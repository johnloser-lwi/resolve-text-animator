# Resolve Text Animator

An OFX plugin that reveals a title one character, word, or line at a time —
Premiere's Text Animator, for DaVinci Resolve.

It works with **no text metadata at all**. The plugin only ever sees rendered
RGBA pixels, and recovers the text structure by measuring pixel distances:
connected-component labeling finds glyphs, a horizontal projection profile finds
lines, and gap statistics find word boundaries. That means it works on any title
generator — Text+, Fusion, or a PNG you drop on the timeline.

## Status

Segmentation, animation and compositing are verified. Word, character and line
grouping are all correct on regular, all-caps, multi-line and script faces.

## Rendering

The plugin renders on the **GPU via CUDA** when Resolve offers it, falling back
to CPU otherwise.

This matters more than it sounds. Resolve's pipeline is GPU-resident, so a
CPU-only OFX plugin forces every frame down to system memory and back — at UHD
that is ~33 MB each way, per frame, stalling the pipeline. That readback, not
the arithmetic, is what stops a CPU plugin playing back in realtime.

Segmentation deliberately stays on the CPU: connected-component labeling with
union-find is a poor fit for GPUs, and it only runs on a cache miss. The result
is one readback per *title change* rather than one per frame. To detect that
change cheaply, the alpha fingerprint is reduced on-device, so the steady-state
per-frame cost off the GPU is an 8-byte readback.

## Requirements

- DaVinci Resolve **Studio** (the free version does not load third-party OFX)
- Windows x64, Visual Studio 18 Community (C++ toolset)
- Optional: CUDA toolkit (12.8+ for RTX 50-series). Without it the plugin still
  builds and runs, CPU-only.

## Build

```powershell
git clone --depth 1 https://github.com/AcademySoftwareFoundation/openfx.git external/openfx
.\build.ps1
```

Produces `build\bundle\TextAnimator.ofx.bundle` and `build\segtest.exe`.

`build.ps1` uses the CMake and Ninja bundled with VS 18 rather than the system
CMake, which predates VS 18 and cannot generate for it.

## Install

```powershell
.\install.ps1
```

Prompts for elevation, copies the bundle to
`C:\Program Files\Common Files\OFX\Plugins`, and reminds you to restart Resolve.
Re-run after each rebuild.

## Use

Put a **Text+ on its own track** with a transparent background, and drop
*OpenFX > Filters > Text > Text Animator* on it.

The title itself must be static — the plugin drives the reveal, so any animation
already on the Text+ will fight it.

Turn on **Show Detection** first. It draws a coloured box around every detected
unit over the dimmed static text, which tells you whether detection is right
independently of whether the timing is right. If words are mis-split, adjust
**Word Gap** (lower splits more eagerly, higher merges more).

### Parameters

| | |
|---|---|
| **Animate By** | Character / Word / Line |
| **Animation** | Fade, or Slide + Fade |
| **Order** | Forward, Reverse, Center Out, Random |
| **Line Order** | Top to Bottom / Bottom to Top. Flips which line goes first while words still read left-to-right within each line — which `Order > Reverse` cannot express, since it reverses those too. |
| **Easing** | Linear, Smoothstep, Cubic Out, Back Out, Custom (see curve editor below) |
| **Start / Duration / Stagger** | seconds; stagger is the delay between consecutive units |
| **Distance Units** | what Slide Distance and Start Blur are measured against: % of Frame Height (default), % of Text Height, or Pixels |
| **Slide Distance / Angle** | how far a unit travels, in the chosen units; 90° rises from below |
| **Start Scale / Start Rotation / Start Blur** | the unit's scale, rotation (degrees, about its own centre) and defocus radius at the beginning of its animation, all unwinding to normal as it settles |
| **Motion Blur** | Motion Blur on/off, Shutter Angle (180° = normal cine shutter), Samples |
| **Detection** | Alpha Threshold, Min Blob Area, Word Gap, Bridge Radius, Show Detection |

### Why distances are ratios

Lengths are authored as a ratio so the motion looks the same at 1080p and 4K.
OFX's `renderScale` cannot do this on its own — it tracks *proxy* scale, not
timeline resolution, so at a 4K timeline it is still 1.0 while the frame and the
text have both doubled. An absolute pixel distance therefore reads as half the
travel it did at 1080p.

*% of Text Height* additionally survives a font-size change, which *% of Frame
Height* does not. *Pixels* is kept for when you want an exact,
resolution-dependent value.

### Curve editor

Set **Easing** to *Custom* and tick **Show Curve Editor**, then set the Viewer's
on-screen-control dropdown to **"Open FX Overlay"** — Resolve hides OFX overlays
until you do. A panel appears top-right of the image: drag the two handles to
shape the easing.

The curve is plotted through the *same evaluator the renderer uses*, so what you
see is exactly what the animation does. A faint diagonal shows linear for
reference, and a yellow dot marks where the current frame sits on the curve.

Handle **x** is clamped to 0..1, because outside it the bezier folds back and one
time would map to two values. **y** is deliberately not clamped — dragging past
the rails is what produces anticipation and overshoot, and the plot range grows
to keep those curves visible. `Curve X1..Y2` stay editable as numbers so a curve
can be typed in or copied between clips.

This lives in the viewer rather than the Inspector because Resolve's OFX host
supports neither parametric (curve) parameters nor custom parameter interacts.

### How the blurs work

Motion blur supersamples each unit's **transform** across the open shutter,
rather than smearing finished pixels along a velocity vector. That is what makes
rotation and scale blur correctly too — there is no single velocity that
describes a spinning glyph. Start Blur rides the same tap loop as a disc-shaped
jitter, so both blurs cost one pass, and `Samples` controls the quality of both.

Taps use a Vogel spiral rather than random offsets, so the blur is stable frame
to frame instead of shimmering.

### Known limits

- **Drop shadows and glows bridge glyphs.** Turn them off on the Text+ and add
  them after, or raise Alpha Threshold.
- **Tightly-joined script fonts** may need Bridge Radius 1–3.
- **Burned-in text** (over footage, no alpha) is not supported — hiding the
  original would require inpainting.

## Testing without Resolve

`segtest` runs the exact segmentation code the plugin uses on a PNG:

```bash
build\segtest.exe testdata\single.png out.png --mode word
```

Writes a diagnostic image with coloured boxes per detected unit. Add
`--strip 8` to render a contact sheet of the animation instead:

```bash
build\segtest.exe testdata\single.png strip.png --mode word --strip 8 --stagger 0.09 --dur 0.45 --slide 50
```

Options: `--mode char|word|line`, `--gap`, `--alpha`, `--bridge`, `--minarea`,
`--strip`, `--stagger`, `--dur`, `--slide`, `--angle`, `--easing`, `--order`,
`--lineorder`, `--rotate`, `--blur`, `--mblur`, `--shutter`, `--samples`, `--fps`.

Note the harness renders with the **CPU** compositor. The CUDA kernels mirror it
but are a separate implementation, so a harness render is not proof the GPU path
agrees — check both when changing the sampling core.

Tune detection here, not in Resolve — the loop is seconds instead of a relaunch.

## Layout

| | |
|---|---|
| `src/segmentation.*` | mask → components → glyphs → lines → words |
| `src/animator.*` | ordering, stagger timing, easing → per-group transform |
| `src/compositor.*` | label-masked sprite extraction and compositing (CPU) |
| `src/cuda_compositor.*` | the same compositing as CUDA kernels, plus on-device hashing |
| `src/diagnostics.*` | detection overlay |
| `src/plugin.cpp` | OFX glue: params, render dispatch |
| `src/interact/` | viewer overlay: draw-suite wrappers and the curve editor |
| `tools/segtest.cpp` | standalone harness |

Everything except `plugin.cpp` is host-independent and operates on a plain float
RGBA view, which is what lets the harness share code with the plugin.
