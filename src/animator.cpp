#include "animator.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace rta {
namespace {

// One coordinate of a cubic bezier from 0 to 1 with the given control values.
inline float bezierAt(float t, float p1, float p2) {
  const float u = 1.0f - t;
  return 3.0f * u * u * t * p1 + 3.0f * u * t * t * p2 + t * t * t;
}

inline float bezierSlope(float t, float p1, float p2) {
  const float u = 1.0f - t;
  return 3.0f * u * u * p1 + 6.0f * u * t * (p2 - p1) + 3.0f * t * t * (1.0f - p2);
}

// Solve x(s) = t for the curve parameter s. The bezier is parameterised by s,
// not by time, so every evaluation needs this inversion first -- the same
// approach browsers use for CSS cubic-bezier().
float solveBezierParam(float t, float x1, float x2) {
  float s = t;  // x is near-linear for sane control points, so t is a good seed

  // Newton converges in a couple of steps when the slope is well conditioned.
  for (int i = 0; i < 8; ++i) {
    const float err = bezierAt(s, x1, x2) - t;
    if (std::fabs(err) < 1e-6f) return s;
    const float d = bezierSlope(s, x1, x2);
    if (std::fabs(d) < 1e-6f) break;  // flat: fall through to bisection
    s -= err / d;
  }

  // Bisection is the safety net for curves with a near-vertical or flat x,
  // where Newton can diverge or stall. Always terminates.
  float lo = 0.0f, hi = 1.0f;
  s = t;
  for (int i = 0; i < 32; ++i) {
    const float x = bezierAt(s, x1, x2);
    if (std::fabs(x - t) < 1e-6f) break;
    if (x < t)
      lo = s;
    else
      hi = s;
    s = 0.5f * (lo + hi);
  }
  return s;
}

}  // namespace

float applyEasing(float t, Easing e, const BezierEasing& b) {
  t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
  switch (e) {
    case Easing::Linear:
      return t;
    case Easing::Smoothstep:
      return t * t * (3.0f - 2.0f * t);
    case Easing::CubicOut: {
      const float u = 1.0f - t;
      return 1.0f - u * u * u;
    }
    case Easing::BackOut: {
      // Overshoots slightly then settles -- the snap that makes a per-word
      // reveal feel deliberate rather than mechanical.
      const float c1 = 1.70158f, c3 = c1 + 1.0f;
      const float u = t - 1.0f;
      return 1.0f + c3 * u * u * u + c1 * u * u;
    }
    case Easing::Custom: {
      const float x1 = std::min(1.0f, std::max(0.0f, b.x1));
      const float x2 = std::min(1.0f, std::max(0.0f, b.x2));
      return bezierAt(solveBezierParam(t, x1, x2), b.y1, b.y2);
    }
  }
  return t;
}

std::vector<int> revealOrder(const std::vector<Group>& groups, int lineCount,
                             const StageSettings& p) {
  const size_t n = groups.size();
  std::vector<int> rank(n);
  if (n == 0) return rank;

  // 1. Reading order, with the line axis optionally flipped.
  std::vector<int> seq(n);
  std::iota(seq.begin(), seq.end(), 0);
  std::stable_sort(seq.begin(), seq.end(), [&](int a, int b) {
    const int la = p.lineOrder == LineOrder::BottomToTop ? lineCount - 1 - groups[a].line
                                                         : groups[a].line;
    const int lb = p.lineOrder == LineOrder::BottomToTop ? lineCount - 1 - groups[b].line
                                                         : groups[b].line;
    if (la != lb) return la < lb;
    return groups[a].indexInLine < groups[b].indexInLine;
  });

  // 2. Permute that sequence.
  switch (p.order) {
    case Order::Forward:
      for (size_t r = 0; r < n; ++r) rank[seq[r]] = int(r);
      break;

    case Order::Reverse:
      for (size_t r = 0; r < n; ++r) rank[seq[r]] = int(n - 1 - r);
      break;

    case Order::CenterOut: {
      std::vector<int> byDist(n);
      std::iota(byDist.begin(), byDist.end(), 0);
      const double mid = (double(n) - 1.0) * 0.5;
      std::stable_sort(byDist.begin(), byDist.end(), [&](int a, int b) {
        return std::fabs(a - mid) < std::fabs(b - mid);
      });
      for (size_t r = 0; r < n; ++r) rank[seq[byDist[r]]] = int(r);
      break;
    }

    case Order::Random: {
      std::vector<int> shuffled(n);
      std::iota(shuffled.begin(), shuffled.end(), 0);
      std::mt19937 rng(uint32_t(p.randomSeed) * 2654435761u + 1u);
      std::shuffle(shuffled.begin(), shuffled.end(), rng);
      for (size_t r = 0; r < n; ++r) rank[seq[r]] = shuffled[r];
      break;
    }
  }
  return rank;
}

namespace {

// The settled pose: on screen, in place, fully opaque.
GroupTransform settledPose() {
  GroupTransform t;
  t.visible = true;
  t.opacity = 1.0f;
  t.scale = 1.0f;
  return t;
}

// The pose at eased value `e`, where 1 is settled and 0 is fully away.
GroupTransform poseAt(float e, const StageSettings& p) {
  GroupTransform t;
  t.visible = true;
  t.opacity = std::min(1.0f, std::max(0.0f, e));
  t.scale = float(p.startScale + (1.0 - p.startScale) * e);
  if (std::fabs(t.scale) < 1e-4f) t.scale = 1e-4f;

  // Rotation and defocus both unwind to zero as the group settles. An
  // overshooting curve pushes e past 1, which is deliberate for rotation (it
  // swings through and back) but meaningless for defocus -- and a negative
  // blur would shrink the group's destination bounds instead of padding them,
  // clipping the word on all four sides.
  t.rotation = float((1.0 - e) * p.startRotation * 3.14159265358979323846 / 180.0);
  t.blur = std::max(0.0f, float((1.0 - e) * p.startBlur));

  if (p.animation == Animation::SlideFade) {
    const double rad = p.slideAngle * 3.14159265358979323846 / 180.0;
    const float d = float((1.0 - e) * p.slideDistance);
    t.offsetX = d * float(std::cos(rad));
    t.offsetY = d * float(std::sin(rad));  // +y is down, so 90deg starts below
  }
  return t;
}

}  // namespace

double stageSpan(size_t groupCount, const StageSettings& s) {
  if (groupCount == 0) return 0.0;
  return double(groupCount - 1) * s.stagger + s.duration;
}

StageSettings mirrorStage(const StageSettings& in, bool mirror) {
  StageSettings out = in;
  if (mirror) return out;  // retreat the way it came: same angle and rotation

  // Continue through: keep travelling in the direction of the entrance, which
  // is the opposite side of the group, and keep turning the same way.
  out.slideAngle = in.slideAngle + 180.0;
  out.startRotation = -in.startRotation;
  return out;
}

GroupTransform transformFor(Stage stage, int revealRank, double frames, double stageStart,
                            const StageSettings& s) {
  if (stage == Stage::Settled) return settledPose();

  const double dur = std::max(1e-6, s.duration);
  const double t0 = stageStart + double(revealRank) * s.stagger;

  // No fudge here. A group's start frame is progress 0 -- opacity 0, fully
  // offset -- and the fade climbs from there. Nudging it so the first frame
  // opened at ~23% made a delayed group pop from nothing straight to a fifth
  // of the way in, which is worse than the blank frame it was meant to fix.
  const float raw = float((frames - t0) / dur);

  if (stage == Stage::In) {
    if (raw <= 0.0f) {
      GroupTransform t;  // not yet started: nothing drawn before the entrance
      t.visible = false;
      return t;
    }
    if (raw >= 1.0f) return settledPose();
    return poseAt(applyEasing(raw, s.easing, s.bezier), s);
  }

  // Exit. Before its window the group is still settled; after it, gone.
  if (raw <= 0.0f) return settledPose();
  if (raw >= 1.0f) {
    GroupTransform t;
    t.visible = false;
    return t;
  }
  // Eased at (1 - raw) rather than (1 - eased(raw)): the exit is the entrance
  // run backwards, so a Cubic Out entrance leaves on the mirrored profile.
  return poseAt(applyEasing(1.0f - raw, s.easing, s.bezier), s);
}

namespace {

// Progress of one stage: 1 is settled, 0 is fully away. An inactive stage sits
// at 1 so it never pulls a group off screen.
//
// `e` may legitimately exceed 1 -- that IS overshoot, and it is what carries a
// group past its resting position before it settles. Whether a group is drawn
// at all is decided by the raw window position, never by `e`, because clamping
// on the eased value throws overshoot away for every property at once.
struct StageEval {
  float e = 1.0f;
  bool hidden = false;
  bool animating = false;  // inside its own window right now
};

StageEval stageEval(Stage stage, int rank, double frames, double stageStart,
                    const StageSettings& s, bool enabled) {
  StageEval r;
  if (!enabled) return r;

  const double dur = std::max(1e-6, s.duration);
  const float raw = float((frames - (stageStart + double(rank) * s.stagger)) / dur);

  if (stage == Stage::In) {
    if (raw <= 0.0f) {
      r.e = 0.0f;
      r.hidden = true;  // not arrived yet
      return r;
    }
    if (raw >= 1.0f) return r;  // settled
    r.e = applyEasing(raw, s.easing, s.bezier);
    r.animating = true;
    return r;
  }

  if (raw <= 0.0f) return r;  // exit not begun
  if (raw >= 1.0f) {
    r.e = 0.0f;
    r.hidden = true;  // gone
    return r;
  }
  r.e = applyEasing(1.0f - raw, s.easing, s.bezier);
  r.animating = true;
  return r;
}

}  // namespace

GroupTransform transformCombined(int rankIn, int rankOut, double frames, double inStart,
                                 double outStart, const StageSettings& in,
                                 const StageSettings& out, bool enableIn, bool enableOut) {
  const StageEval a = stageEval(Stage::In, rankIn, frames, inStart, in, enableIn);
  const StageEval b = stageEval(Stage::Out, rankOut, frames, outStart, out, enableOut);

  // Drawn or not is decided by the windows, so an overshooting curve keeps its
  // overshoot instead of being flattened to the resting pose.
  if (a.hidden || b.hidden) {
    GroupTransform t;  // not arrived, or gone
    t.visible = false;
    return t;
  }

  // The stage that is actually running owns the group, and brings its own
  // slide, rotation and blur with it -- so an exit that interrupts an entrance
  // departs using the exit's settings rather than inheriting the entrance's.
  //
  // Whether it is running comes from its window, NOT from comparing eased
  // values. An overshooting entrance sits above 1 while the dormant exit rests
  // at exactly 1, so "further from settled" handed the group to the stage doing
  // nothing and flattened the overshoot away.
  //
  // When both really are running -- a genuine overlap -- the smaller value wins,
  // which is the one further from arrived.
  const bool outLeads = b.animating && (!a.animating || b.e < a.e);
  // poseAt is the identity at exactly 1, so a settled group needs no special
  // case, and a value past 1 comes through as overshoot.
  return poseAt(outLeads ? b.e : a.e, outLeads ? out : in);
}

void transformTapsCombined(int rankIn, int rankOut, double frames, double inStart, double outStart,
                           const AnimParams& p, const StageSettings& in, const StageSettings& out,
                           GroupTransform* outTaps) {
  const int n = tapCount(p);
  if (n == 1) {
    outTaps[0] = transformCombined(rankIn, rankOut, frames, inStart, outStart, in, out,
                                   p.enableIn, p.enableOut);
    return;
  }
  const double span = p.motionBlur ? (p.shutterAngle / 360.0) : 0.0;
  for (int k = 0; k < n; ++k) {
    const double f = (double(k) / double(n - 1)) - 0.5;
    outTaps[k] = transformCombined(rankIn, rankOut, frames + span * f, inStart, outStart, in, out,
                                   p.enableIn, p.enableOut);
  }
}

FitReport checkFit(size_t groupCount, const AnimParams& p, double clipLength, bool haveLength) {
  FitReport r;
  r.inNeeds = p.startTime + stageSpan(groupCount, p.in);
  if (!haveLength || groupCount == 0) return r;

  // The entrance must settle before the clip ends.
  if (p.enableIn) r.inFits = r.inNeeds <= clipLength;

  // The exit must begin after the clip starts, or its first groups are already
  // part-gone on the clip's first frame.
  if (p.enableOut) {
    r.outStart = clipLength - p.outOffset - stageSpan(groupCount, p.out);
    r.outFits = r.outStart >= 0.0;
  }
  return r;
}

int adaptiveTapCount(const std::vector<Group>& groups, const std::vector<GroupTransform>& ends,
                     int maxTaps, double pixelsPerSample) {
  if (maxTaps <= 1) return 1;
  if (ends.size() < groups.size() * 2) return maxTaps;  // no measurement: pay full

  const double perSample = std::max(0.25, pixelsPerSample);

  double maxTravel = 0.0;  // pixels a group moves across the shutter
  double maxBlur = 0.0;    // largest defocus radius, either end

  for (size_t g = 0; g < groups.size(); ++g) {
    const GroupTransform& a = ends[g * 2];
    const GroupTransform& b = ends[g * 2 + 1];
    if (!a.visible && !b.visible) continue;

    maxBlur = std::max({maxBlur, double(a.blur), double(b.blur)});

    // A fade with no movement still varies across the shutter, so opacity has
    // to count as travel or a cross-fading group would collapse to one tap and
    // band. Scaled into pixels so it shares the one budget: a full 0->1 fade is
    // worth as much as crossing the group's own height.
    //
    // A tap outside the group's window contributes nothing to the composite, so
    // it counts as opacity 0 rather than as its default-constructed pose.
    const RectI& bb = groups[g].bbox;
    const double opA = a.visible ? double(a.opacity) : 0.0;
    const double opB = b.visible ? double(b.opacity) : 0.0;
    maxTravel = std::max(maxTravel, std::fabs(opB - opA) * double(bb.height()));

    // Corner travel needs two real poses. Measuring against a tap that draws
    // nothing compares against an identity pose that was never on screen, and
    // reads the group's whole start offset as travel -- which spiked the budget
    // to 47 taps on the frame a group first appears, for a group that is barely
    // visible yet.
    if (!a.visible || !b.visible) continue;

    // Corner travel, which unlike a centre offset also catches a group that
    // spins or scales without going anywhere.
    const float cx = 0.5f * float(bb.x1 + bb.x2);
    const float cy = 0.5f * float(bb.y1 + bb.y2);
    const float xs[4] = {float(bb.x1), float(bb.x2), float(bb.x1), float(bb.x2)};
    const float ys[4] = {float(bb.y1), float(bb.y1), float(bb.y2), float(bb.y2)};
    for (int i = 0; i < 4; ++i) {
      double px[2], py[2];
      const GroupTransform* t[2] = {&a, &b};
      for (int e = 0; e < 2; ++e) {
        const double c = std::cos(t[e]->rotation) * t[e]->scale;
        const double s = std::sin(t[e]->rotation) * t[e]->scale;
        const double dx = xs[i] - cx, dy = ys[i] - cy;
        px[e] = cx + c * dx - s * dy + t[e]->offsetX;
        py[e] = cy + s * dx + c * dy + t[e]->offsetY;
      }
      maxTravel = std::max(maxTravel, std::hypot(px[1] - px[0], py[1] - py[0]));
    }
  }

  // Motion: one tap per `perSample` pixels of travel, so the smear is sampled
  // at roughly constant density however fast the group is going.
  int need = int(maxTravel / perSample) + 1;

  // Defocus: the taps tile a disc rather than a line, so the count that holds
  // the same spacing goes with the AREA. Treating it as linear leaves visible
  // rings in a large blur.
  if (maxBlur > 0.0) {
    const double perAxis = 2.0 * maxBlur / perSample;
    need = std::max(need, int(perAxis * perAxis * 0.25) + 1);
    // A disc sampled once is not a blur, it is the image nudged off centre by
    // the single Vogel offset. Any defocus at all buys at least two taps.
    need = std::max(need, 2);
  }

  return std::min(maxTaps, std::max(1, need));
}

int tapCount(const AnimParams& p) {
  // Defocus alone still needs multiple taps; only a fully sharp, non-blurred
  // render can get away with one.
  const bool needsTaps = p.motionBlur || p.in.startBlur > 0.0 || p.out.startBlur > 0.0;
  if (!needsTaps) return 1;
  return std::min(64, std::max(2, p.blurSamples));
}

void transformTaps(Stage stage, int revealRank, double frames, double stageStart,
                   const AnimParams& p, const StageSettings& s, GroupTransform* out) {
  const int n = tapCount(p);
  if (n == 1) {
    out[0] = transformFor(stage, revealRank, frames, stageStart, s);
    return;
  }

  // Spread the taps across the open shutter, centred on the frame time. Time is
  // in frames, so one frame is 1.0 and the shutter is simply its fraction --
  // no frame rate needed.
  const double span = p.motionBlur ? (p.shutterAngle / 360.0) : 0.0;
  for (int k = 0; k < n; ++k) {
    const double f = (double(k) / double(n - 1)) - 0.5;
    out[k] = transformFor(stage, revealRank, frames + span * f, stageStart, s);
  }
}

void discOffset(int tap, int taps, float* dx, float* dy) {
  // Vogel spiral: even coverage of the disc without an RNG, so the blur is
  // stable frame to frame instead of shimmering.
  const float golden = 2.39996322972865332f;
  const float a = float(tap) * golden;
  const float r = std::sqrt((float(tap) + 0.5f) / float(std::max(1, taps)));
  *dx = r * std::cos(a);
  *dy = r * std::sin(a);
}

double totalDuration(size_t groupCount, const AnimParams& p) {
  if (groupCount == 0) return 0.0;
  return p.startTime + stageSpan(groupCount, p.in);
}

}  // namespace rta
