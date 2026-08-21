#include "timeline_widget.h"

#include <algorithm>
#include <cmath>

#include "../edit_block.h"
#include "../warning_state.h"

namespace rta {
namespace {

constexpr double kLaneHeightPx = 16.0;
constexpr double kLaneGapPx = 4.0;
constexpr double kHeaderPx = 18.0;
constexpr double kPadPx = 8.0;
constexpr double kLabelPx = 30.0;  // room for the lane names at the left

// Five frames is a useful beat at both 24 and 30 fps -- about a fifth of a
// second either way -- and still a nudge rather than a jump.
constexpr double kCoarseStepFrames = 5.0;

constexpr Colour kIn{0.20f, 0.72f, 1.00f, 0.85f};
constexpr Colour kOut{1.00f, 0.55f, 0.35f, 0.85f};
constexpr Colour kOverlap{1.00f, 0.25f, 0.25f, 0.95f};
constexpr Colour kPlayhead{1.00f, 0.85f, 0.25f, 1.00f};
constexpr Colour kText{0.92f, 0.92f, 0.92f, 1.00f};
constexpr Colour kTextDim{0.60f, 0.60f, 0.60f, 1.00f};
constexpr Colour kGrid{1.00f, 1.00f, 1.00f, 0.18f};

}  // namespace

void TimelineWidget::layout(const OverlayContext& c) {
  const double lanes = 2.0;
  const double height =
      c.sy(kHeaderPx + kPadPx * 2.0 + lanes * kLaneHeightPx + (lanes - 1.0) * kLaneGapPx);

  // Centred along the bottom, wide enough to read but not so wide it swamps
  // the picture. Same placement as Multi Transform, so the two feel alike.
  const double rodW = c.rod.x2 - c.rod.x1;
  double width = rodW * 0.66;
  const double minWidth = c.sx(360.0);
  if (width < minWidth) width = std::min(minWidth, rodW * 0.95);

  _rect.x1 = c.rod.x1 + (rodW - width) * 0.5;
  _rect.x2 = _rect.x1 + width;
  _rect.y1 = c.rod.y1 + c.sy(20.0);
  _rect.y2 = _rect.y1 + height;

  const AnalysisState a = analysisState(c.effect);

  // The clip is the frame of reference: the ruler runs 0..clip length, so a
  // bar reads as "this far into the clip" rather than as a raw host frame.
  double t0 = 0.0, t1 = 0.0;
  if (a.haveLength && a.clipLength > 0.0) {
    t1 = a.clipLength;
  } else {
    // No clip extent: frame whatever there is.
    t0 = a.inStart;
    t1 = a.inEnd;
  }
  // A bar dragged outside the clip still has to stay visible and grabbable.
  t0 = std::min(t0, a.inStart);
  t1 = std::max(t1, a.inEnd);
  if (a.enableOut && a.outUsable) {
    t0 = std::min(t0, a.outStart);
    t1 = std::max(t1, a.outEnd);
  }
  t0 = std::min(t0, a.clipFrame);
  t1 = std::max(t1, a.clipFrame);

  double span = t1 - t0;
  if (span < 1.0) span = 1.0;
  _t0 = t0 - span * 0.12;
  _t1 = t1 + span * 0.12;

  if (_drag != kNone) {
    _t0 = _dragT0;
    _t1 = _dragT1;
  }
}

double TimelineWidget::frameToX(double frame) const {
  const double span = (_t1 - _t0) > 1e-9 ? (_t1 - _t0) : 1.0;
  return _rect.x1 + (frame - _t0) / span * (_rect.x2 - _rect.x1);
}

double TimelineWidget::xToFrame(double x) const {
  const double w = (_rect.x2 - _rect.x1) > 1e-9 ? (_rect.x2 - _rect.x1) : 1.0;
  return _t0 + (x - _rect.x1) / w * (_t1 - _t0);
}

OfxRectD TimelineWidget::laneRect(const OverlayContext& c, int lane) const {
  const double laneH = c.sy(kLaneHeightPx);
  const double gap = c.sy(kLaneGapPx);
  const double top = _rect.y2 - c.sy(kHeaderPx + kPadPx);
  OfxRectD r;
  r.y2 = top - double(lane) * (laneH + gap);
  r.y1 = r.y2 - laneH;
  r.x1 = _rect.x1 + c.sx(kPadPx + kLabelPx);
  r.x2 = _rect.x2 - c.sx(kPadPx);
  return r;
}

void TimelineWidget::draw(const OverlayContext& c) {
  const AnalysisState a = analysisState(c.effect);
  Panel(c, _rect);

  SetColour(c, kTextDim);
  Text(c, "TIMING", _rect.x1 + c.sx(kPadPx), _rect.y2 - c.sy(6.0),
       kOfxDrawTextAlignmentLeft | kOfxDrawTextAlignmentTop);

  if (!a.haveTiming) {
    SetColour(c, kTextDim);
    Text(c, "no frame rendered yet", _rect.x1 + c.sx(kPadPx + kLabelPx),
         (_rect.y1 + _rect.y2) * 0.5, kOfxDrawTextAlignmentLeft | kOfxDrawTextAlignmentTop);
    return;
  }

  SetColour(c, kText);
  Text(c, "frame " + FrameLabel(a.clipFrame), _rect.x2 - c.sx(kPadPx), _rect.y2 - c.sy(6.0),
       kOfxDrawTextAlignmentRight | kOfxDrawTextAlignmentTop);

  // The clip extent, so it is obvious where frame 0 and the last frame are,
  // and whether a bar has been pushed outside them.
  const double top = _rect.y2 - c.sy(kHeaderPx);
  if (a.haveLength && a.clipLength > 0.0) {
    const double cx0 = frameToX(0.0), cx1 = frameToX(a.clipLength);
    SetColour(c, Colour{1.0f, 1.0f, 1.0f, 0.08f});
    FillRect(c, cx0, _rect.y1 + c.sy(3.0), cx1, top);
    SetColour(c, kTextDim);
    SetLineWidth(c, 1.0f);
    Line(c, cx0, _rect.y1 + c.sy(3.0), cx0, top);
    Line(c, cx1, _rect.y1 + c.sy(3.0), cx1, top);
    Text(c, "0", cx0 + c.sx(3.0), _rect.y1 + c.sy(4.0),
         kOfxDrawTextAlignmentLeft | kOfxDrawTextAlignmentBottom);
    Text(c, FrameLabel(a.clipLength), cx1 - c.sx(3.0), _rect.y1 + c.sy(4.0),
         kOfxDrawTextAlignmentRight | kOfxDrawTextAlignmentBottom);
  }

  struct Lane {
    const char* name;
    bool on;
    double s, e;
    Colour col;
  };
  const Lane lanes[2] = {
      {"IN", true, a.inStart, a.inEnd, kIn},
      {"OUT", a.enableOut && a.outUsable, a.outStart, a.outEnd, kOut},
  };

  // Where the two bars share frames. Drawn on both lanes, so the overlap reads
  // as one red band across the strip rather than as a detail of either bar.
  const bool overlap = lanes[1].on && lanes[1].s < lanes[0].e && lanes[0].s < lanes[1].e;
  const double ovA = std::max(lanes[0].s, lanes[1].s);
  const double ovB = std::min(lanes[0].e, lanes[1].e);

  for (int i = 0; i < 2; ++i) {
    const OfxRectD lane = laneRect(c, i);
    const Lane& L = lanes[i];

    SetColour(c, kGrid);
    SetLineWidth(c, 1.0f);
    Line(c, lane.x1, lane.y1 - c.sy(1.0), lane.x2, lane.y1 - c.sy(1.0));

    SetColour(c, L.on ? L.col : kTextDim);
    Text(c, L.name, _rect.x1 + c.sx(kPadPx), (lane.y1 + lane.y2) * 0.5 + c.sy(4.0),
         kOfxDrawTextAlignmentLeft | kOfxDrawTextAlignmentTop);

    if (!L.on) {
      SetColour(c, kTextDim);
      const char* why = (i == 1 && a.enableOut && !a.outUsable) ? "no clip end" : "off";
      Text(c, why, lane.x1 + c.sx(4.0), (lane.y1 + lane.y2) * 0.5 + c.sy(4.0),
           kOfxDrawTextAlignmentLeft | kOfxDrawTextAlignmentTop);
      continue;
    }

    double bx1 = frameToX(L.s), bx2 = frameToX(L.e);
    if (bx2 - bx1 < c.sx(3.0)) bx2 = bx1 + c.sx(3.0);

    SetColour(c, L.col);
    FillRect(c, bx1, lane.y1, bx2, lane.y2);
    if (overlap) {
      SetColour(c, kOverlap);
      FillRect(c, frameToX(ovA), lane.y1, frameToX(ovB), lane.y2);
    }

    SetColour(c, kText);
    Text(c, FrameLabel(L.s), bx1 - c.sx(4.0), (lane.y1 + lane.y2) * 0.5 + c.sy(4.0),
         kOfxDrawTextAlignmentRight | kOfxDrawTextAlignmentTop);
    Text(c, FrameLabel(L.e), bx2 + c.sx(4.0), (lane.y1 + lane.y2) * 0.5 + c.sy(4.0),
         kOfxDrawTextAlignmentLeft | kOfxDrawTextAlignmentTop);
  }

  if (overlap) {
    SetColour(c, kOverlap);
    Text(c, "overlap " + FrameLabel(ovB - ovA) + " f", _rect.x1 + c.sx(kPadPx + 60.0),
         _rect.y2 - c.sy(6.0), kOfxDrawTextAlignmentLeft | kOfxDrawTextAlignmentTop);
  }

  // Playhead last, so it reads on top of the lanes.
  const double px = frameToX(a.clipFrame);
  if (px >= _rect.x1 && px <= _rect.x2) {
    SetColour(c, kPlayhead);
    SetLineWidth(c, 1.5f);
    Line(c, px, _rect.y1 + c.sy(3.0), px, top);
  }
}

bool TimelineWidget::penDown(const OverlayContext& c, const OfxPointD& p) {
  if (!Contains(_rect, p)) return false;
  const AnalysisState a = analysisState(c.effect);
  if (!a.haveTiming) return true;

  for (int i = 0; i < 2; ++i) {
    const OfxRectD lane = laneRect(c, i);
    if (p.y < lane.y1 || p.y > lane.y2) continue;
    const bool on = i == 0 || (a.enableOut && a.outUsable);
    if (!on) return true;

    const double s = i == 0 ? a.inStart : a.outStart;
    const double e = i == 0 ? a.inEnd : a.outEnd;
    double bx1 = frameToX(s), bx2 = frameToX(e);
    if (bx2 - bx1 < c.sx(3.0)) bx2 = bx1 + c.sx(3.0);
    if (p.x < bx1 - c.sx(4.0) || p.x > bx2 + c.sx(4.0)) return true;

    try {
      _grabValue = c.effect->fetchDoubleParam(i == 0 ? "startTime" : "outOffset")->getValue();
    } catch (...) {
      return true;
    }
    _drag = i == 0 ? kDragIn : kDragOut;
    _grabFrame = xToFrame(p.x);
    _dragT0 = _t0;
    _dragT1 = _t1;
    beginEdit(c.effect, i == 0 ? "Drag entrance timing" : "Drag exit timing");
    _editing = true;
    return true;
  }
  return true;  // clicks inside the panel never fall through to the image
}

bool TimelineWidget::penMotion(const OverlayContext& c, const OfxPointD& p) {
  if (_drag == kNone) return false;
  const double delta = xToFrame(p.x) - _grabFrame;
  // Snapping the movement, not the value, so a bar keeps whatever offset it
  // was authored with and simply slides in whole-frame or five-frame steps.
  const double step = c.shiftHeld ? kCoarseStepFrames : 1.0;
  const double shift = SnapTo(delta, step);
  try {
    if (_drag == kDragIn) {
      c.effect->fetchDoubleParam("startTime")->setValue(_grabValue + shift);
    } else {
      // Out Offset counts BACK from the clip end, so dragging the bar later
      // means a smaller offset. Not clamped at zero: the parameter allows a
      // negative offset, and the bar should be able to go wherever it can.
      c.effect->fetchDoubleParam("outOffset")->setValue(_grabValue - shift);
    }
  } catch (...) {
  }
  return true;
}

bool TimelineWidget::penUp(const OverlayContext& c, const OfxPointD&) {
  if (_drag == kNone) return false;
  _drag = kNone;
  if (_editing) {
    endEdit(c.effect);
    _editing = false;
  }
  return true;
}

void TimelineWidget::abandon(const OverlayContext& c) {
  _drag = kNone;
  if (_editing) {
    endEdit(c.effect);
    _editing = false;
  }
}

}  // namespace rta
