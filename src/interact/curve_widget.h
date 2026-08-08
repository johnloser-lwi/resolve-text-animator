// On-screen cubic-bezier easing editor.
//
// This lives in the viewer rather than the Inspector because Resolve's OFX host
// supports neither parametric (curve) parameters nor custom parameter
// interacts. The viewer overlay is the only surface available for a real curve
// editor -- and it is arguably the better one, since you watch the animation
// while you shape it.
//
// Remember Resolve hides OFX overlays until the on-screen-control dropdown
// under the Viewer is set to "Open FX Overlay".
#pragma once

#include "draw_utils.h"

namespace rta {

class CurveWidget {
 public:
  void layout(const OverlayContext& c);
  void draw(const OverlayContext& c);

  bool penDown(const OverlayContext& c, const OfxPointD& p);
  bool penMotion(const OverlayContext& c, const OfxPointD& p);
  bool penUp(const OverlayContext& c, const OfxPointD& p);

  bool dragging() const { return _drag != kNone; }

 private:
  enum Drag { kNone, kDragP1, kDragP2 };

  // Grows the plotted y range so an overshooting curve stays visible instead of
  // being drawn flat against the panel edge.
  void fitRange(const OverlayContext& c);

  OfxPointD unitToPanel(double ux, double uy) const;
  OfxPointD panelToUnit(const OfxPointD& p) const;
  void writeHandle(const OverlayContext& c, int which, const OfxPointD& unit);

  OfxRectD _rect{0, 0, 0, 0};  // whole panel
  OfxRectD _plot{0, 0, 0, 0};  // plotting area inside it
  double _yMin = 0.0, _yMax = 1.0;
  Drag _drag = kNone;
};

}  // namespace rta
