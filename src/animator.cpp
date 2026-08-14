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

std::vector<int> readingSequence(const std::vector<Group>& groups, int lineCount,
                                 LineOrder lineOrder) {
  std::vector<int> seq(groups.size());
  std::iota(seq.begin(), seq.end(), 0);
  std::stable_sort(seq.begin(), seq.end(), [&](int a, int b) {
    const int la =
        lineOrder == LineOrder::BottomToTop ? lineCount - 1 - groups[a].line : groups[a].line;
    const int lb =
        lineOrder == LineOrder::BottomToTop ? lineCount - 1 - groups[b].line : groups[b].line;
    if (la != lb) return la < lb;
    return groups[a].indexInLine < groups[b].indexInLine;
  });
  return seq;
}

void applyRevealRange(const std::vector<int>& rank, int first, int last, bool isolate, int taps,
                      std::vector<GroupTransform>* transforms) {
  taps = std::max(1, taps);
  if (!transforms || transforms->size() != rank.size() * size_t(taps)) return;
  // 1 and 0 are "from the beginning" and "to the end", so the untouched defaults
  // cover the whole frame and cost nothing.
  if (first <= 1 && last <= 0) return;

  for (size_t g = 0; g < rank.size(); ++g) {
    const int n = rank[g] + 1;  // the numbers count from one

    if (n < first) {
      if (isolate) {
        // Nothing but the range, so the element can be judged on its own.
        for (int k = 0; k < taps; ++k)
          (*transforms)[g * size_t(taps) + size_t(k)].visible = false;
        continue;
      }
      // Already introduced on an earlier clip, so it simply sits there. Settled
      // is a pose, not a stage: identity transform, fully opaque.
      for (int k = 0; k < taps; ++k) {
        GroupTransform t;
        t.visible = true;
        (*transforms)[g * size_t(taps) + size_t(k)] = t;
      }
    } else if (last > 0 && n > last) {
      // Not introduced yet. Hidden rather than transparent, so it costs nothing
      // to draw and cannot leave a faint edge behind.
      for (int k = 0; k < taps; ++k)
        (*transforms)[g * size_t(taps) + size_t(k)].visible = false;
    }
  }
}

std::vector<int> revealOrder(const std::vector<Group>& groups, int lineCount,
                             const StageSettings& p) {
  const size_t n = groups.size();
  std::vector<int> rank(n);
  if (n == 0) return rank;

  // 1. Reading order, with the line axis optionally flipped.
  const std::vector<int> seq = readingSequence(groups, lineCount, p.lineOrder);

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

    case Order::Manual: {
      // Named units first, in the order named; everything else keeps reading
      // order behind them. Partial lists are the point -- pulling one element to
      // the front should not mean writing out the whole sequence.
      std::vector<char> placed(n, 0);
      int next = 0;
      for (int ci : p.manualOrder) {
        for (size_t g = 0; g < n; ++g) {
          if (placed[g]) continue;
          const std::vector<int>& idx = groups[g].glyphIndices;
          if (std::find(idx.begin(), idx.end(), ci) == idx.end()) continue;
          rank[g] = next++;
          placed[g] = 1;
          break;  // a number names one unit; repeating it does nothing
        }
      }
      // seq already honours lineOrder, so the remainder reads the way the rest
      // of the plugin means by "in order".
      for (size_t r = 0; r < n; ++r) {
        const int g = seq[r];
        if (placed[size_t(g)]) continue;
        rank[g] = next++;
        placed[size_t(g)] = 1;
      }
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

std::vector<double> revealDelays(const std::vector<Group>& groups, const std::vector<int>& rank,
                                 const StageSettings& s) {
  const size_t n = groups.size();
  std::vector<double> delays(n, 0.0);
  if (n == 0 || rank.size() != n) return delays;

  if (!s.staggerByLength) {
    for (size_t i = 0; i < n; ++i) delays[i] = double(rank[i]);
    return delays;
  }

  // Walk the reveal order and accumulate characters. Each group waits for all
  // the text before it, so a long word pushes everything after it back by its
  // own length rather than by a single step.
  std::vector<size_t> byRank(n);
  for (size_t i = 0; i < n; ++i) {
    const int r = rank[i];
    if (r >= 0 && size_t(r) < n) byRank[size_t(r)] = i;
  }
  double acc = 0.0;
  for (size_t r = 0; r < n; ++r) {
    const size_t gi = byRank[r];
    delays[gi] = acc;
    acc += double(std::max(1, groups[gi].glyphCount));
  }
  return delays;
}

double maxDelay(const std::vector<double>& delays) {
  double m = 0.0;
  for (double d : delays) m = std::max(m, d);
  return m;
}

double stageSpan(double maxDelayUnits, const StageSettings& s) {
  return std::max(0.0, maxDelayUnits) * s.stagger + s.duration;
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

GroupTransform transformFor(Stage stage, double delayUnits, double frames, double stageStart,
                            const StageSettings& s) {
  if (stage == Stage::Settled) return settledPose();

  const double dur = std::max(1e-6, s.duration);
  const double t0 = stageStart + delayUnits * s.stagger;

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

StageEval stageEval(Stage stage, double delayUnits, double frames, double stageStart,
                    const StageSettings& s, bool enabled) {
  StageEval r;
  if (!enabled) return r;

  const double dur = std::max(1e-6, s.duration);
  const float raw = float((frames - (stageStart + delayUnits * s.stagger)) / dur);

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

GroupTransform transformCombined(double delayIn, double delayOut, double frames, double inStart,
                                 double outStart, const StageSettings& in,
                                 const StageSettings& out, bool enableIn, bool enableOut) {
  const StageEval a = stageEval(Stage::In, delayIn, frames, inStart, in, enableIn);
  const StageEval b = stageEval(Stage::Out, delayOut, frames, outStart, out, enableOut);

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

void transformTapsCombined(double delayIn, double delayOut, double frames, double inStart,
                           double outStart, const AnimParams& p, const StageSettings& in,
                           const StageSettings& out, GroupTransform* outTaps) {
  const int n = tapCount(p);
  if (n == 1) {
    outTaps[0] = transformCombined(delayIn, delayOut, frames, inStart, outStart, in, out,
                                   p.enableIn, p.enableOut);
    return;
  }
  const double span = p.motionBlur ? (p.shutterAngle / 360.0) : 0.0;
  for (int k = 0; k < n; ++k) {
    const double f = (double(k) / double(n - 1)) - 0.5;
    outTaps[k] = transformCombined(delayIn, delayOut, frames + span * f, inStart, outStart, in, out,
                                   p.enableIn, p.enableOut);
  }
}

FitReport checkFit(double maxDelayIn, double maxDelayOut, const AnimParams& p, double clipLength,
                   bool haveLength) {
  FitReport r;
  r.inNeeds = p.startTime + stageSpan(maxDelayIn, p.in);
  if (!haveLength) return r;

  // The entrance must settle before the clip ends.
  if (p.enableIn) r.inFits = r.inNeeds <= clipLength;

  // The exit must begin after the clip starts, or its first groups are already
  // part-gone on the clip's first frame.
  if (p.enableOut) {
    r.outStart = clipLength - p.outOffset - stageSpan(maxDelayOut, p.out);
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

  for (size_t g = 0; g < groups.size(); ++g) {
    const GroupTransform& a = ends[g * 2];
    const GroupTransform& b = ends[g * 2 + 1];
    if (!a.visible && !b.visible) continue;

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

  // Defocus no longer buys taps. It used to demand them by AREA -- the samples
  // tiled a disc, so holding the spacing meant squaring the count, and a wide
  // blur alone could eat the whole budget. The compositor blurs the sprite
  // directly now, so the taps are free to measure only how far things move.
  return std::min(maxTaps, std::max(1, need));
}

int tapCount(const AnimParams& p) {
  // Taps are for MOTION only now. Defocus used to need them too -- it was N
  // jittered copies of the sprite -- so a still, defocused group paid for a
  // sample count it had no movement to spend it on, and looked like a row of
  // ghosts for the trouble. The compositor blurs the finished sprite instead,
  // at a cost that does not depend on the radius.
  if (!p.motionBlur) return 1;
  return std::min(64, std::max(2, p.blurSamples));
}

void transformTaps(Stage stage, double delayUnits, double frames, double stageStart,
                   const AnimParams& p, const StageSettings& s, GroupTransform* out) {
  const int n = tapCount(p);
  if (n == 1) {
    out[0] = transformFor(stage, delayUnits, frames, stageStart, s);
    return;
  }

  // Spread the taps across the open shutter, centred on the frame time. Time is
  // in frames, so one frame is 1.0 and the shutter is simply its fraction --
  // no frame rate needed.
  const double span = p.motionBlur ? (p.shutterAngle / 360.0) : 0.0;
  for (int k = 0; k < n; ++k) {
    const double f = (double(k) / double(n - 1)) - 0.5;
    out[k] = transformFor(stage, delayUnits, frames + span * f, stageStart, s);
  }
}

double totalDuration(double maxDelayUnits, const AnimParams& p) {
  return p.startTime + stageSpan(maxDelayUnits, p.in);
}

std::vector<float> spacingOffsets(const Segmentation& seg, const SpacingParams& sp) {
  std::vector<float> d(seg.groups.size() * 2, 0.0f);
  // Letter height is what makes these settings survive a change of resolution;
  // without a measurement there is nothing to be a fraction OF, so do nothing
  // rather than guess a pixel count.
  if (!sp.any() || seg.letterHeight <= 0.0f) return d;
  const float h = seg.letterHeight;
  const int lines = std::max(1, seg.lineCount);

  // How many units each line holds, so an anchor can be named as a position
  // within the line instead of as a pixel.
  std::vector<int> perLine(size_t(lines), 0);
  for (const Group& g : seg.groups)
    if (g.line >= 0 && size_t(g.line) < perLine.size())
      perLine[size_t(g.line)] = std::max(perLine[size_t(g.line)], g.indexInLine + 1);

  float lineRef = 0.5f * float(lines - 1);
  if (sp.lineAnchor == LineAnchor::Top) lineRef = 0.0f;
  if (sp.lineAnchor == LineAnchor::Bottom) lineRef = float(lines - 1);

  for (size_t i = 0; i < seg.groups.size(); ++i) {
    const Group& g = seg.groups[i];
    const int n =
        (g.line >= 0 && size_t(g.line) < perLine.size()) ? std::max(1, perLine[size_t(g.line)]) : 1;

    // The unit that stays put. Everything else moves away from it in proportion
    // to how many units separate them -- which is what tracking IS, a constant
    // added to every advance.
    float ref = 0.5f * float(n - 1);
    if (sp.trackAnchor == TrackAnchor::Left) ref = 0.0f;
    if (sp.trackAnchor == TrackAnchor::Right) ref = float(n - 1);

    d[i * 2 + 0] = (float(g.indexInLine) - ref) * sp.tracking * h;
    d[i * 2 + 1] = (float(g.line) - lineRef) * sp.lineSpacing * h;
  }
  return d;
}

std::vector<int> mapUnitsToGroups(const Segmentation& unitSeg, const Segmentation& groupSeg) {
  std::vector<int> out(unitSeg.groups.size(), 0);
  if (groupSeg.groups.empty()) return out;

  int maxIndex = 0;
  for (const Group& g : groupSeg.groups)
    for (int ci : g.glyphIndices) maxIndex = std::max(maxIndex, ci);
  for (const Group& g : unitSeg.groups)
    for (int ci : g.glyphIndices) maxIndex = std::max(maxIndex, ci);

  std::vector<int> byChar(size_t(maxIndex) + 1, -1);
  for (size_t gi = 0; gi < groupSeg.groups.size(); ++gi)
    for (int ci : groupSeg.groups[gi].glyphIndices)
      if (ci >= 0 && size_t(ci) < byChar.size()) byChar[size_t(ci)] = int(gi);

  for (size_t u = 0; u < unitSeg.groups.size(); ++u) {
    const Group& g = unitSeg.groups[u];
    int found = -1;
    for (int ci : g.glyphIndices) {
      if (ci >= 0 && size_t(ci) < byChar.size() && byChar[size_t(ci)] >= 0) {
        found = byChar[size_t(ci)];
        break;
      }
    }
    // A unit whose characters are in no group cannot be animated by one; parking
    // it on the first group keeps it on screen rather than dropping it silently.
    out[u] = found >= 0 ? found : 0;
  }
  return out;
}

void applySpacing(const Segmentation& unitSeg, const Segmentation& groupSeg,
                  const std::vector<int>& unitToGroup, const std::vector<float>& offsets,
                  const std::vector<GroupTransform>& groupTaps, int taps,
                  std::vector<GroupTransform>* outTaps, std::vector<float>* outPivots) {
  taps = std::max(1, taps);
  const size_t nUnits = unitSeg.groups.size(), nGroups = groupSeg.groups.size();
  outTaps->assign(nUnits * size_t(taps), GroupTransform{});
  outPivots->assign(nUnits * 2, 0.0f);
  if (nUnits == 0 || nGroups == 0) return;

  // Each group's centre once the offsets have moved its units. Averaged over the
  // units, which for an unmoved group is exactly its old bbox centre.
  std::vector<float> sum(nGroups * 2, 0.0f);
  std::vector<int> count(nGroups, 0);
  for (size_t u = 0; u < nUnits; ++u) {
    const size_t g = size_t(unitToGroup[u]);
    if (g >= nGroups) continue;
    sum[g * 2 + 0] += offsets.size() > u * 2 + 1 ? offsets[u * 2 + 0] : 0.0f;
    sum[g * 2 + 1] += offsets.size() > u * 2 + 1 ? offsets[u * 2 + 1] : 0.0f;
    ++count[g];
  }
  std::vector<float> pivot(nGroups * 2, 0.0f);
  for (size_t g = 0; g < nGroups; ++g) {
    const RectI& b = groupSeg.groups[g].bbox;
    const float n = float(std::max(1, count[g]));
    pivot[g * 2 + 0] = 0.5f * float(b.x1 + b.x2) + sum[g * 2 + 0] / n;
    pivot[g * 2 + 1] = 0.5f * float(b.y1 + b.y2) + sum[g * 2 + 1] / n;
  }

  for (size_t u = 0; u < nUnits; ++u) {
    const size_t g = size_t(unitToGroup[u]);
    if (g >= nGroups) continue;
    const float dx = offsets.size() > u * 2 + 1 ? offsets[u * 2 + 0] : 0.0f;
    const float dy = offsets.size() > u * 2 + 1 ? offsets[u * 2 + 1] : 0.0f;
    (*outPivots)[u * 2 + 0] = pivot[g * 2 + 0];
    (*outPivots)[u * 2 + 1] = pivot[g * 2 + 1];

    for (int k = 0; k < taps; ++k) {
      GroupTransform t = groupTaps[g * size_t(taps) + size_t(k)];
      const float c = std::cos(t.rotation) * t.scale;
      const float s = std::sin(t.rotation) * t.scale;
      t.offsetX += c * dx - s * dy;
      t.offsetY += s * dx + c * dy;
      (*outTaps)[u * size_t(taps) + size_t(k)] = t;
    }
  }
}

}  // namespace rta
