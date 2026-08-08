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

  // The +1 makes t0 the first ANIMATING frame rather than a dead one.
  //
  // Without it, raw is exactly 0 on the group's own start frame, which reads as
  // "not started yet" and draws nothing -- so the clip's first frame is blank
  // and the reveal only appears to begin on the second. It also stretched a
  // duration of N across N+1 frames.
  //
  // With it, a 12-frame duration occupies frames t0 .. t0+11: the first shows
  // the opening step of the fade, and the last lands exactly on settled.
  const float raw = float((frames - t0 + 1.0) / dur);

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
