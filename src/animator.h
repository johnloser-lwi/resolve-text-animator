// Turns a set of detected groups plus a time into a per-group transform.
// Pure math -- no pixels, no host.
#pragma once

#include <vector>

#include "segmentation.h"

namespace rta {

enum class Animation { Fade = 0, SlideFade = 1 };
enum class Easing { Linear = 0, Smoothstep = 1, CubicOut = 2, BackOut = 3, Custom = 4 };
enum class Order { Forward = 0, Reverse = 1, CenterOut = 2, Random = 3 };
enum class LineOrder { TopToBottom = 0, BottomToTop = 1 };

// Control points of the Custom easing curve, as a CSS-style cubic bezier from
// (0,0) to (1,1). x is clamped to 0..1 so the curve stays a function of time;
// y is deliberately unclamped, since overshoot above 1 and anticipation below 0
// are exactly what make a reveal feel snappy.
struct BezierEasing {
  float x1 = 0.25f, y1 = 0.1f, x2 = 0.25f, y2 = 1.0f;  // ~ CSS "ease"
};

// Everything that describes how one stage -- the entrance or the exit -- moves.
// Both stages take the same shape so the exit can simply copy the entrance when
// they are linked.
struct StageSettings {
  Animation animation = Animation::SlideFade;
  Easing easing = Easing::CubicOut;
  BezierEasing bezier;
  Order order = Order::Forward;
  LineOrder lineOrder = LineOrder::TopToBottom;
  int randomSeed = 0;

  // Timing is in FRAMES. Frames rather than seconds because compositing is
  // authored in frames, and because it removes the need to ask the host for a
  // frame rate at all -- Fusion does not publish one on its clips, and asking
  // for it there threw out of the render action and failed every frame.
  double duration = 12.0;  // frames, per group
  double stagger = 2.0;    // frames between consecutive groups

  double slideDistance = 40.0;  // pixels, already resolution-scaled by caller
  double slideAngle = 90.0;     // degrees; 90 == rises from below
  double startScale = 1.0;
  double startRotation = 0.0;  // degrees away from settled
  double startBlur = 0.0;      // pixels of defocus away from settled
};

// Which stage is driving a given frame.
//
// A frame is only ever in one stage. Settled is the quiet middle where the
// entrance has finished and the exit has not begun -- and also what you get
// with both stages switched off, where the text simply sits there.
enum class Stage { In, Out, Settled };

struct AnimParams {
  bool enableIn = true;
  bool enableOut = false;

  StageSettings in;
  StageSettings out;

  double startTime = 0.0;  // frames after the clip start before the entrance begins
  // Frames before the clip's END at which the exit *finishes*. Anchoring the
  // exit to the end rather than the start is the whole point: it stays put when
  // the clip is trimmed or its length changes.
  double outOffset = 0.0;

  bool motionBlur = false;
  double shutterAngle = 180.0;  // degrees of a frame the shutter is open
  int blurSamples = 8;          // taps for motion blur and defocus alike
};

struct GroupTransform {
  float offsetX = 0.0f;
  float offsetY = 0.0f;
  float scale = 1.0f;
  float rotation = 0.0f;  // radians, about the group's own centre
  float blur = 0.0f;      // defocus radius in pixels
  float opacity = 1.0f;
  bool visible = false;
};

// Reveal position of each group, indexed by group.
//
// Reading order is established first (honouring lineOrder), then the order mode
// permutes that sequence. Keeping the two separate is what lets a title reveal
// bottom line first while each line still reads left to right -- which
// Order::Reverse alone cannot express, since it also reverses within the line.
std::vector<int> revealOrder(const std::vector<Group>& groups, int lineCount,
                             const StageSettings& s);

// Frames from the first group starting to the last one finishing.
double stageSpan(size_t groupCount, const StageSettings& s);

// Pose of one group.
//
// `stageStart` is the frame the stage's first group begins on: startTime for
// the entrance, and (clipLength - outOffset - stageSpan) for the exit.
//
// The exit is the entrance played backwards -- eased at (1 - raw) rather than
// (1 - eased(raw)) -- so a Cubic Out entrance leaves with the mirrored profile
// instead of an unrelated one.
GroupTransform transformFor(Stage stage, int revealRank, double frames, double stageStart,
                            const StageSettings& s);

// How many transform samples each group needs this frame. 1 unless motion blur
// or a defocus is in play, in which case the shutter interval is supersampled.
int tapCount(const AnimParams& p);

// Fills `out` with tapCount(p) transforms spanning the shutter interval.
//
// Sampling the transform itself, rather than smearing the finished pixels along
// a velocity vector, is what makes rotation and scale blur correctly too --
// there is no single velocity that describes a spinning glyph.
void transformTaps(Stage stage, int revealRank, double frames, double stageStart,
                   const AnimParams& p, const StageSettings& s, GroupTransform* out);

// Derives the exit settings from the entrance when the two are linked.
//
// Mirror means the text retreats the way it came: an entrance that rises from
// below sinks back down, which is the same slide angle, because the exit runs
// the transform backwards. Without mirror the exit *continues* in the direction
// of travel, which is that angle turned through 180 degrees.
StageSettings mirrorStage(const StageSettings& in, bool mirror);

// Deterministic unit-disc offset for defocus sampling (Vogel spiral).
void discOffset(int tap, int taps, float* dx, float* dy);

// The easing evaluator. Exposed so the on-screen curve editor plots the exact
// curve the renderer uses, rather than its own approximation of it.
float applyEasing(float t, Easing e, const BezierEasing& b);

// Total time from startTime until the last group has settled.
double totalDuration(size_t groupCount, const AnimParams& p);

}  // namespace rta
