// Two-lane timing strip: the entrance and the exit, as bars on a ruler of the
// clip, with the playhead over both.
//
// Brought over from Multi Transform, where the lanes are what make staggered
// stages legible. Here there are only ever two, but the question they answer
// is the one the Inspector cannot: do the entrance and the exit OVERLAP, and by
// how much. Start, Duration, Stagger and Out Offset are four numbers in two
// groups; on the strip they are two bars, and an overlap is two bars touching.
//
// The bars are read from what the renderer published, not recomputed here: a
// span depends on how many units detection found and in what order, and the
// overlay has no business running detection. That also means the strip shows
// exactly the timing that was drawn, never an estimate of it.
#pragma once

#include "draw_utils.h"

namespace rta {

class TimelineWidget {
 public:
  void layout(const OverlayContext& c);
  void draw(const OverlayContext& c);

  bool penDown(const OverlayContext& c, const OfxPointD& p);
  bool penMotion(const OverlayContext& c, const OfxPointD& p);
  bool penUp(const OverlayContext& c, const OfxPointD& p);

  bool dragging() const { return _drag != kNone; }

  // Drop a drag whose mouse-up never arrived, closing the edit block it left
  // open. Safe to call when nothing is in progress.
  void abandon(const OverlayContext& c);

  const OfxRectD& rect() const { return _rect; }

 private:
  // Whole-bar drags only. The entrance bar moves Start; the exit bar moves Out
  // Offset. Neither end is grabbable on its own, because a bar's length here is
  // not a parameter -- it is Duration plus Stagger times the unit count -- and
  // a drag that silently rewrote Duration would be the wrong kind of surprise.
  enum Drag { kNone, kDragIn, kDragOut };

  OfxRectD laneRect(const OverlayContext& c, int lane) const;
  double frameToX(double frame) const;
  double xToFrame(double x) const;

  OfxRectD _rect{0, 0, 0, 0};
  double _t0 = 0.0, _t1 = 1.0;  // visible clip-frame range

  Drag _drag = kNone;
  bool _editing = false;
  double _grabFrame = 0.0;   // clip frame under the cursor at pen-down
  double _grabValue = 0.0;   // the parameter's value at pen-down
  // The visible range is frozen for the duration of a drag: recomputing it live
  // would move the bar out from under the cursor as it is dragged.
  double _dragT0 = 0.0, _dragT1 = 1.0;
};

}  // namespace rta
