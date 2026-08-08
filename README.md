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
| **In Animation / Out Animation** | enable each stage independently; both off leaves the text static |
| **Animate By** | Character / Word / Line |
| **Animation** | Fade, or Slide + Fade |
| **Order** | Forward, Reverse, Center Out, Random |
| **Line Order** | Top to Bottom / Bottom to Top. Flips which line goes first while words still read left-to-right within each line — which `Order > Reverse` cannot express, since it reverses those too. |
| **Easing** | Linear, Smoothstep, Cubic Out, Back Out, Custom (see curve editor below) |
| **Start / Duration / Stagger** | **frames**; Start is measured from the clip's first frame, Stagger is the delay between consecutive units (fractional values allowed) |
| **Distance Units** | what Slide Distance and Start Blur are measured against: % of Frame Height (default), % of Text Height, or Pixels |
| **Slide Distance / Angle** | how far a unit travels, in the chosen units; 90° rises from below |
| **Start Scale / Start Rotation / Start Blur** | the unit's scale, rotation (degrees, about its own centre) and defocus radius at the beginning of its animation, all unwinding to normal as it settles |
| **Motion Blur** | Motion Blur on/off, Shutter Angle (180° = normal cine shutter), Samples |
| **Out Animation** (group) | End Offset (frames before the clip end at which it finishes), Link to In, Mirror, Clip Length Override, plus a full independent set used when unlinked |
| **Detection** | Alpha Threshold, Min Blob Area, Word Gap, Bridge Radius, Show Detection |

### In and out animations

**In Animation** and **Out Animation** toggle independently. With both off the
text simply sits there, unanimated.

The exit is anchored to the **end** of the clip, not the start: `End Offset = 30`
means the text is fully gone 30 frames before the clip ends. Anchoring it to the
end is the point — it stays where you put it when the clip is trimmed or its
length changes.

**Link to In** (default on) makes the exit reuse the entrance's settings, and
**Mirror** decides how:

| Mirror | Behaviour |
|---|---|
| On (default) | The text **retreats the way it came** — an entrance rising from below sinks back down. |
| Off | The text **continues** in its direction of travel and carries on upward. |

Untick **Link to In** and the exit gets its own full set of controls, including
its own grouping — words in, characters out, if you like. The two stages are
segmented separately and a frame is only ever driven by one of them, which is
what makes differing group modes possible at all.

The exit is the entrance run backwards — eased at `1 - raw` rather than
`1 - eased(raw)` — so a Cubic Out entrance leaves on the mirrored profile rather
than an unrelated one.

Because the exit needs to know where the clip ends, it is skipped when the host
reports no usable clip length. **Clip Length Override** covers that case.

### Timing is in frames, measured from the clip start

**Start** is measured from the clip's own first frame, so `0` means "when the
clip starts", not timeline frame zero. Move or trim the clip and the animation
travels with it — trimming the head re-anchors the reveal to the new first frame
rather than stranding it at a fixed timeline position.

Timing is authored in **frames** rather than seconds. That suits compositing,
and it has a second benefit that matters more: nothing ever has to ask the host
for a frame rate. Fusion does not publish `kOfxImageEffectPropFrameRate` on its
clips, and the Support library *throws* when a property is missing — thrown out
of the render action that becomes `kOfxStatErrMissingHostFeature`, failing every
frame so the effect appears to do nothing at all. Working in frames removes the
dependency instead of guarding it. Motion blur benefits too: the shutter is
simply a fraction of `1.0`, with no conversion involved.

#### Set Start to Playhead

**Resolve does not tell a plugin where a trimmed clip visually starts.** Every
route reports the *available media* instead, so the answer shifts by the length
of the unused head handle. Measured on Resolve Studio 21, on a clip visibly
running frames 1000→1149:

| Source | Reported | Correct? |
|---|---|---|
| `getFrameRange()` | 1000-minute sentinel | no |
| `getUnmappedFrameRange` | `[0, 0]` | no |
| `getEffectDuration()` | `184` (available span, not the 150 visible) | no |
| `timeLineGetBounds` t1 | `967` (start **minus** head handle) | no |
| `timeLineGetBounds` t2 | `1149` — held still across trims | **the end only** |

Drag the head back by one frame and t1 jumped 83 frames, because that is how
much unused media sat behind it. Only the end is anchored to the clip.

So the start is captured rather than derived: park the playhead on the clip's
first frame and press **Set Start to Playhead**. That frame becomes animation
frame 0. Re-press it after re-trimming the head. Untick **Use Manual Start** to
fall back to the host's reported start, which is exact only on clips with no
head handle.

The exit needs the clip *end*, which `timeLineGetBounds` does report reliably,
so it needs no such button.

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
