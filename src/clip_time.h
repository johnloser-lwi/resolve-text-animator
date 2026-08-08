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
inline bool getClipRange(OFX::ImageEffect* effect, double& outStart, double& outLength) {
  if (!effect) return false;
  double t1 = 0.0, t2 = 0.0;
  try {
    effect->timeLineGetBounds(t1, t2);
  } catch (...) {
    return false;
  }
  return validateClipRange(t1, t2, outStart, outLength);
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
