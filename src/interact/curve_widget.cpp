#include "curve_widget.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "edit_block.h"

namespace rta {
namespace {

constexpr double kPanelPx = 190.0;
constexpr double kPadPx = 10.0;
constexpr double kHeaderPx = 16.0;
constexpr int kCurveSegments = 96;
constexpr double kYMinDefault = 0.0;
constexpr double kYMaxDefault = 1.0;

double clampd(double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); }

const char* easingName(Easing e) {
  switch (e) {
    case Easing::Linear: return "LINEAR";
    case Easing::Smoothstep: return "SMOOTHSTEP";
    case Easing::CubicOut: return "CUBIC OUT";
    case Easing::BackOut: return "BACK OUT";
    case Easing::Custom: return "CUSTOM";
  }
  return "";
}

}  // namespace

void CurveWidget::fitRange(const OverlayContext& c) {
  // A Back Out or a hand-shaped overshoot swings well past 0..1, and a curve
  // clipped to the panel edge tells you nothing about its shape.
  double lo = kYMinDefault, hi = kYMaxDefault;
  for (int i = 0; i <= kCurveSegments; ++i) {
    const double y = applyEasing(float(i) / kCurveSegments, c.anim.easing, c.anim.bezier);
    lo = std::min(lo, y);
    hi = std::max(hi, y);
  }

  // Include the handles themselves: they are draggable, so they must stay
  // reachable even when they sit outside the curve's own extent.
  if (c.anim.easing == Easing::Custom) {
    lo = std::min({lo, double(c.anim.bezier.y1), double(c.anim.bezier.y2)});
    hi = std::max({hi, double(c.anim.bezier.y1), double(c.anim.bezier.y2)});
  }

  const double margin = 0.12 * std::max(1.0, hi - lo);
  _yMin = std::min(kYMinDefault, lo - margin);
  _yMax = std::max(kYMaxDefault, hi + margin);
}

void CurveWidget::layout(const OverlayContext& c) {
  fitRange(c);

  const double size = c.sx(kPanelPx);

  // Top-right of the image, inset far enough not to fight the safe-area guides.
  _rect.x2 = c.rod.x2 - c.sx(20.0);
  _rect.x1 = _rect.x2 - size;
  _rect.y2 = c.rod.y2 - c.sy(20.0);
  _rect.y1 = _rect.y2 - (size + c.sy(kHeaderPx));

  _plot.x1 = _rect.x1 + c.sx(kPadPx);
  _plot.x2 = _rect.x2 - c.sx(kPadPx);
  _plot.y1 = _rect.y1 + c.sy(kPadPx);
  _plot.y2 = _rect.y2 - c.sy(kPadPx + kHeaderPx);
}

OfxPointD CurveWidget::unitToPanel(double ux, double uy) const {
  OfxPointD p;
  p.x = _plot.x1 + ux * (_plot.x2 - _plot.x1);
  p.y = _plot.y1 + (uy - _yMin) / (_yMax - _yMin) * (_plot.y2 - _plot.y1);
  return p;
}

OfxPointD CurveWidget::panelToUnit(const OfxPointD& p) const {
  const double w = (_plot.x2 - _plot.x1) > 1e-9 ? (_plot.x2 - _plot.x1) : 1.0;
  const double h = (_plot.y2 - _plot.y1) > 1e-9 ? (_plot.y2 - _plot.y1) : 1.0;
  OfxPointD u;
  u.x = (p.x - _plot.x1) / w;
  u.y = _yMin + (p.y - _plot.y1) / h * (_yMax - _yMin);
  return u;
}

void CurveWidget::draw(const OverlayContext& c) {
  Panel(c, _rect);

  const BezierEasing& b = c.anim.bezier;
  const bool custom = c.anim.easing == Easing::Custom;

  SetColour(c, colours::kTextDim);
  Text(c, easingName(c.anim.easing), _rect.x1 + c.sx(kPadPx), _rect.y2 - c.sy(5.0),
       kOfxDrawTextAlignmentLeft | kOfxDrawTextAlignmentTop);

  // Reference frame: 0 and 1 bound the useful range, so anything drawn outside
  // them is anticipation or overshoot.
  const OfxPointD zero0 = unitToPanel(0.0, 0.0);
  const OfxPointD zero1 = unitToPanel(1.0, 0.0);
  const OfxPointD one0 = unitToPanel(0.0, 1.0);
  const OfxPointD one1 = unitToPanel(1.0, 1.0);

  SetColour(c, colours::kGrid);
  SetLineWidth(c, 1.0f);
  Line(c, zero0.x, zero0.y, zero1.x, zero1.y);
  Line(c, one0.x, one0.y, one1.x, one1.y);
  StrokeRect(c, _plot.x1, _plot.y1, _plot.x2, _plot.y2);

  // A faint diagonal shows what linear would look like, which makes the amount
  // of easing legible at a glance.
  SetColour(c, colours::kGuide);
  Line(c, zero0.x, zero0.y, one1.x, one1.y);

  // The curve, plotted through the same evaluator the renderer uses -- so what
  // is drawn here is exactly what the animation does.
  OfxPointD pts[kCurveSegments + 1];
  for (int i = 0; i <= kCurveSegments; ++i) {
    const double u = double(i) / kCurveSegments;
    pts[i] = unitToPanel(u, applyEasing(float(u), c.anim.easing, b));
  }
  SetColour(c, colours::kAccent);
  SetLineWidth(c, 2.0f);
  Polyline(c, pts, kCurveSegments + 1);

  if (custom) {
    // Handles, tethered to the endpoints they belong to.
    const OfxPointD p0 = unitToPanel(0.0, 0.0);
    const OfxPointD p3 = unitToPanel(1.0, 1.0);
    const OfxPointD p1 = unitToPanel(b.x1, b.y1);
    const OfxPointD p2 = unitToPanel(b.x2, b.y2);

    SetColour(c, colours::kTether);
    SetLineWidth(c, 1.0f);
    Line(c, p0.x, p0.y, p1.x, p1.y);
    Line(c, p3.x, p3.y, p2.x, p2.y);

    Handle(c, p1.x, p1.y, _drag == kDragP1 ? colours::kPlayhead : colours::kHandle, 4.5);
    Handle(c, p2.x, p2.y, _drag == kDragP2 ? colours::kPlayhead : colours::kHandle, 4.5);
  }

  // Playhead: where the current frame sits on the curve.
  if (c.hasProgress) {
    const double raw = clampd(c.progress, 0.0, 1.0);
    const OfxPointD ph = unitToPanel(raw, applyEasing(float(raw), c.anim.easing, b));
    SetColour(c, colours::kPlayhead);
    Ellipse(c, ph.x, ph.y, c.sx(4.0), c.sy(4.0));
  }

  // Numeric read-out: the curve is for feel, the numbers for repeatability.
  char buf[96];
  if (custom)
    std::snprintf(buf, sizeof(buf), "%.2f %.2f  %.2f %.2f", b.x1, b.y1, b.x2, b.y2);
  else
    std::snprintf(buf, sizeof(buf), "set Easing to Custom");
  SetColour(c, custom ? colours::kText : colours::kTextDim);
  Text(c, buf, _rect.x2 - c.sx(kPadPx), _rect.y2 - c.sy(5.0),
       kOfxDrawTextAlignmentRight | kOfxDrawTextAlignmentTop);
}

void CurveWidget::writeHandle(const OverlayContext& c, int which, const OfxPointD& unit) {
  // x is clamped to 0..1 because outside it the bezier folds back on itself and
  // one time would map to two values, which the solver cannot answer. Every
  // curve editor constrains this, Resolve's included.
  //
  // y is deliberately NOT clamped to 0..1: dragging a handle past the rails is
  // exactly what produces anticipation and overshoot, and clamping would make
  // half the useful curve space unreachable.
  const double ux = clampd(unit.x, 0.0, 1.0);
  const double uy = clampd(unit.y, -10.0, 10.0);

  if (which == kDragP1) {
    c.effect->fetchDoubleParam("easeX1")->setValue(ux);
    c.effect->fetchDoubleParam("easeY1")->setValue(uy);
  } else {
    c.effect->fetchDoubleParam("easeX2")->setValue(ux);
    c.effect->fetchDoubleParam("easeY2")->setValue(uy);
  }
}

bool CurveWidget::penDown(const OverlayContext& c, const OfxPointD& p) {
  if (!Contains(_rect, p)) return false;
  // Swallow clicks anywhere in the panel so a miss near a handle does not fall
  // through and start some other drag underneath.
  if (c.anim.easing != Easing::Custom) return true;

  const BezierEasing& b = c.anim.bezier;
  const OfxPointD p1 = unitToPanel(b.x1, b.y1);
  const OfxPointD p2 = unitToPanel(b.x2, b.y2);

  if (NearPoint(c, p, p1.x, p1.y, 10.0))
    _drag = kDragP1;
  else if (NearPoint(c, p, p2.x, p2.y, 10.0))
    _drag = kDragP2;
  else
    _drag = kNone;

  if (_drag != kNone) {
    // One block for the whole drag, so the host sees a single edit and a single
    // undo step rather than one per mouse-move.
    beginEdit(c.effect, "Adjust easing curve");
    _editing = true;
    writeHandle(c, _drag, panelToUnit(p));
  }
  return true;
}

bool CurveWidget::penMotion(const OverlayContext& c, const OfxPointD& p) {
  if (_drag == kNone) return false;
  writeHandle(c, _drag, panelToUnit(p));
  return true;
}

bool CurveWidget::penUp(const OverlayContext& c, const OfxPointD&) {
  if (_drag == kNone) return false;
  _drag = kNone;
  if (_editing) {
    _editing = false;
    endEdit(c.effect);
  }
  return true;
}

void CurveWidget::abandon(const OverlayContext& c) {
  _drag = kNone;
  if (!_editing) return;
  _editing = false;
  forceCloseEdits(c.effect);
}

}  // namespace rta
