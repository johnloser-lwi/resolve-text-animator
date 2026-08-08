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

struct AnimParams {
  Animation animation = Animation::SlideFade;
  Easing easing = Easing::CubicOut;
  BezierEasing bezier;
  Order order = Order::Forward;
  LineOrder lineOrder = LineOrder::TopToBottom;
  int randomSeed = 0;
  // All timing is in FRAMES, measured from the clip's first frame.
  //
  // Frames rather than seconds because compositing is authored in frames, and
  // because it removes the need to ask the host for a frame rate at all --
  // Fusion does not publish one on its clips, and asking for it there threw out
  // of the render action and failed every frame.
  double startTime = 0.0;   // frames before the first group appears
  double duration = 12.0;   // frames, per group
  double stagger = 2.0;     // frames between consecutive groups
  double slideDistance = 40.0;  // pixels at analysis resolution
  double slideAngle = 90.0;     // degrees; 90 == rises from below
  double startScale = 1.0;
  double startRotation = 0.0;   // degrees the group is rotated by at p=0
  double startBlur = 0.0;       // pixels of defocus at p=0, ramping to sharp

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
std::vector<int> revealOrder(const std::vector<Group>& groups, int lineCount, const AnimParams& p);

GroupTransform transformFor(int revealRank, double time, const AnimParams& p);

// How many transform samples each group needs this frame. 1 unless motion blur
// is on, in which case the shutter interval is supersampled.
int tapCount(const AnimParams& p);

// Fills `out` with tapCount(p) transforms spanning the shutter interval.
//
// Sampling the transform itself, rather than smearing the finished pixels along
// a velocity vector, is what makes rotation and scale blur correctly too --
// there is no single velocity that describes a spinning glyph.
void transformTaps(int revealRank, double time, const AnimParams& p, GroupTransform* out);

// Deterministic unit-disc offset for defocus sampling (Vogel spiral).
void discOffset(int tap, int taps, float* dx, float* dy);

// The easing evaluator. Exposed so the on-screen curve editor plots the exact
// curve the renderer uses, rather than its own approximation of it.
float applyEasing(float t, Easing e, const BezierEasing& b);

// Total time from startTime until the last group has settled.
double totalDuration(size_t groupCount, const AnimParams& p);

}  // namespace rta
