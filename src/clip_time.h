// Where the clip starts, and how to express a render time relative to it.
//
// Timing is authored clip-relative -- Start 0 means the clip's first frame --
// so that moving or trimming a clip carries the animation with it instead of
// leaving it stranded at a fixed timeline position. Converting between the two
// needs exactly one number: where the clip starts.
//
// Finding that number takes measuring, because most of the obvious routes are
// dead ends in Resolve (values observed on Resolve Studio, a 155-frame clip on
// a timeline starting at 01:00:00:00):
//
//   src/dst getFrameRange          [0, 1798200]      a sentinel, exactly 1000 minutes
//   src/dst getUnmappedFrameRange  [0, 0]            not populated
//   timeLineGetTime                0                 not populated
//   timeLineGetBounds              [107961, 108116]  <- the clip
//
// Only the last is usable, hence the validation below: it exists to tell a
// genuine answer apart from a host that is really saying "I do not know".
#pragma once

#include "ofxsImageEffect.h"

namespace rta {

// There is deliberately no frame-rate helper here.
//
// All timing is authored in frames, so nothing ever needs to ask the host for
// kOfxImageEffectPropFrameRate. That matters: Fusion does not publish it on
// clips, and the Support library *throws* when a property is missing. Thrown
// out of the render action it becomes kOfxStatErrMissingHostFeature, failing
// every single frame -- the effect simply appears to do nothing. Resolve's Edit
// page does publish it, which is why an unguarded read worked there for a long
// time and only fell over once the plugin was used in Fusion.
//
// Working in frames removes the dependency rather than guarding it.

// Decide whether a reported [t1, t2] is a real clip extent.
inline bool validateClipRange(double t1, double t2, double& outStart, double& outLength) {
  // NaN fails every comparison, so reject it explicitly rather than hoping the
  // range checks below happen to catch it.
  if (!(t1 == t1) || !(t2 == t2)) return false;

  const double span = t2 - t1;
  if (!(span > 0.0)) return false;   // a real clip is at least one frame
  if (span > 1.0e6) return false;    // ...and nowhere near 1000 minutes
  outStart = t1;
  outLength = span;
  return true;
}

// The clip's extent in timeline frames. False if the host reports nothing
// usable, in which case the caller should treat render times as already
// clip-relative.
//
// The reported extent includes the clip's HEAD HANDLE, so once a clip's head is
// dragged back the start can run before the beginning of the timeline and comes
// back negative. Resolve never renders those frames -- render times are clamped
// at zero -- so taking the reported start literally shifts every animation
// later by the whole length of the handle, and an entrance looks already
// half-finished on the clip's first visible frame.
//
// Measured on Resolve Studio 21, dragging one clip's head progressively back:
//
//   start=18   len=187   rendered args.time 18..205     start + len = 205
//   start=43   len=254   rendered args.time 43..195     start + len = 297
//   start=-66  len=363   rendered args.time  0..53      start + len = 297
//   start=-79  len=376   rendered args.time  0..297     start + len = 297
//
// The end (start + len) is stable and the start is not, which is what gives the
// game away: only the end is anchored to the clip. The first frame that will
// actually be rendered is therefore max(0, start), and the usable length is
// measured from there.

// The raw, validated bounds exactly as the host reports them.
//
// t1 is NOT the clip's visible start: it includes the unused head handle, so it
// moves whenever the available media changes. t2 is the only end of this range
// that stayed put across trims in testing. Callers wanting an animation origin
// want the manual anchor instead; this is here for the clip END.
inline bool getClipBounds(OFX::ImageEffect* effect, double& t1, double& t2) {
  if (!effect) return false;
  double a = 0.0, b = 0.0;
  try {
    effect->timeLineGetBounds(a, b);
  } catch (...) {
    return false;
  }
  double lo = 0.0, span = 0.0;
  if (!validateClipRange(a, b, lo, span)) return false;
  t1 = a;
  t2 = b;
  return true;
}

inline bool getClipRange(OFX::ImageEffect* effect, double& outStart, double& outLength) {
  if (!effect) return false;
  double t1 = 0.0, t2 = 0.0;
  try {
    effect->timeLineGetBounds(t1, t2);
  } catch (...) {
    return false;
  }

  // t1 is used as-is, and deliberately NOT clamped to zero.
  //
  // It is the earliest the clip could reach -- it includes the unused head
  // handle -- so it is not the visible start, and no API reports that (see the
  // manual anchor). But it has the one property that matters for robustness:
  // t1 is always at or before the earliest frame the host will ask for, so the
  // clip-relative time is never negative and the animation always plays.
  //
  // Clamping it to zero breaks exactly that. Resolve really does render
  // negative timeline frames once a clip is dragged so it begins before the
  // start of the timeline -- measured args.time down to -112 against
  // raw=[-463, 0] -- and a clamped origin turns those into negative clip
  // frames, where an entrance has not started yet and so draws nothing at all.
  //
  // The exit is unaffected either way: origin + length == t2, so it lands on
  // the same absolute frame whatever the origin is.
  outStart = t1;
  outLength = t2 - t1;
  return outLength > 0.0;
}

// Convert a render time to clip-relative frames, where 0 is the clip's first
// frame. Falls through unchanged when the extent is unknown.
inline double toClipTime(OFX::ImageEffect* effect, double absoluteTime) {
  double start = 0.0, length = 0.0;
  if (!getClipRange(effect, start, length)) return absoluteTime;

  // Guard against a host that already hands out clip-relative render times: it
  // would report bounds near 108000 while passing a time near 0, and blindly
  // subtracting would push the animation tens of minutes into the past. A
  // genuine absolute time always sits at or after the clip's start, give or
  // take the clip's own length.
  if (absoluteTime < start - length) return absoluteTime;

  return absoluteTime - start;
}

}  // namespace rta
