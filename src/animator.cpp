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
                             const AnimParams& p) {
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

GroupTransform transformFor(int revealRank, double time, const AnimParams& p) {
  GroupTransform t;

  const double dur = std::max(1e-6, p.duration);
  const double t0 = p.startTime + double(revealRank) * p.stagger;
  const float raw = float((time - t0) / dur);

  if (raw <= 0.0f) {
    // Not yet started: fully hidden, so nothing is drawn before startTime.
    t.visible = false;
    return t;
  }
  t.visible = true;

  if (raw >= 1.0f) {
    t.opacity = 1.0f;
    t.scale = 1.0f;
    return t;
  }

  const float e = applyEasing(raw, p.easing, p.bezier);
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

int tapCount(const AnimParams& p) {
  // Defocus alone still needs multiple taps; only a fully sharp, non-blurred
  // render can get away with one.
  const bool needsTaps = p.motionBlur || p.startBlur > 0.0;
  if (!needsTaps) return 1;
  return std::min(64, std::max(2, p.blurSamples));
}

void transformTaps(int revealRank, double time, const AnimParams& p, GroupTransform* out) {
  const int n = tapCount(p);
  if (n == 1) {
    out[0] = transformFor(revealRank, time, p);
    return;
  }

  // Spread the taps across the open shutter, centred on the frame time.
  const double span = p.motionBlur ? p.frameDuration * (p.shutterAngle / 360.0) : 0.0;
  for (int k = 0; k < n; ++k) {
    const double f = (double(k) / double(n - 1)) - 0.5;
    out[k] = transformFor(revealRank, time + span * f, p);
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
  return p.startTime + double(groupCount - 1) * p.stagger + p.duration;
}

}  // namespace rta
