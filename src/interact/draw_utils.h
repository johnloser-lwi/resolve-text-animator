// Thin helpers over OfxDrawSuiteV1.
//
// Everything the overlay draws is in *canonical image coordinates*, because
// that is the space the host sets up the projection for. Anything that should
// look a fixed size on screen regardless of viewer zoom has to be scaled by
// pixelScale, the size of one screen pixel in canonical units. Getting that
// wrong is the classic overlay bug where handles become unusably tiny as soon
// as you zoom out.
#pragma once

#include <cmath>
#include <cstdio>
#include <string>

#include "ofxDrawSuite.h"
#include "ofxsImageEffect.h"

#include "animator.h"

namespace rta {

struct Colour {
  float r, g, b, a;
};

// Restrained on purpose: the overlay sits on top of the user's picture, so it
// has to stay legible without shouting over the image.
namespace colours {
constexpr Colour kPanel = {0.08f, 0.09f, 0.11f, 0.82f};
constexpr Colour kPanelEdge = {0.35f, 0.38f, 0.44f, 0.90f};
constexpr Colour kText = {0.88f, 0.90f, 0.94f, 1.00f};
constexpr Colour kTextDim = {0.55f, 0.58f, 0.64f, 1.00f};
constexpr Colour kGrid = {0.28f, 0.30f, 0.35f, 0.70f};
constexpr Colour kGuide = {0.35f, 0.38f, 0.44f, 0.55f};
constexpr Colour kAccent = {0.20f, 0.72f, 1.00f, 1.00f};
constexpr Colour kPlayhead = {1.00f, 0.85f, 0.25f, 1.00f};
constexpr Colour kHandle = {1.00f, 1.00f, 1.00f, 1.00f};
constexpr Colour kTether = {0.85f, 0.88f, 0.95f, 0.55f};
}  // namespace colours

// Everything a widget needs to draw itself and interpret a click.
struct OverlayContext {
  OFX::ImageEffect* effect = nullptr;
  OfxDrawSuiteV1* draw = nullptr;
  OfxDrawContextHandle ctx = nullptr;

  double time = 0.0;        // host time of this event
  double progress = 0.0;    // 0..1 through the whole reveal, for the playhead
  bool hasProgress = false;

  OfxPointD pixelScale{1.0, 1.0};
  OfxRectD rod{0.0, 0.0, 0.0, 0.0};  // image bounds, canonical

  // Shift, tracked from key events because PenArgs does not carry modifiers.
  // Coarsens a timeline drag to five-frame steps.
  bool shiftHeld = false;

  // The animation exactly as the renderer will see it, read straight from the
  // parameters, so the overlay can never show something different.
  AnimParams anim;

  double sx(double screenPixels) const { return screenPixels * pixelScale.x; }
  double sy(double screenPixels) const { return screenPixels * pixelScale.y; }
};

// --- primitives -------------------------------------------------------------

inline void SetColour(const OverlayContext& c, const Colour& col) {
  const OfxRGBAColourF v = {col.r, col.g, col.b, col.a};
  c.draw->setColour(c.ctx, &v);
}

inline void SetLineWidth(const OverlayContext& c, float w) { c.draw->setLineWidth(c.ctx, w); }

inline void FillRect(const OverlayContext& c, double x1, double y1, double x2, double y2) {
  const OfxPointD p[2] = {{x1, y1}, {x2, y2}};
  c.draw->draw(c.ctx, kOfxDrawPrimitiveRectangle, p, 2);
}

inline void StrokeRect(const OverlayContext& c, double x1, double y1, double x2, double y2) {
  const OfxPointD p[4] = {{x1, y1}, {x2, y1}, {x2, y2}, {x1, y2}};
  c.draw->draw(c.ctx, kOfxDrawPrimitiveLineLoop, p, 4);
}

inline void Line(const OverlayContext& c, double x1, double y1, double x2, double y2) {
  const OfxPointD p[2] = {{x1, y1}, {x2, y2}};
  c.draw->draw(c.ctx, kOfxDrawPrimitiveLines, p, 2);
}

inline void Polyline(const OverlayContext& c, const OfxPointD* pts, int n) {
  if (n >= 2) c.draw->draw(c.ctx, kOfxDrawPrimitiveLineStrip, pts, n);
}

inline void Ellipse(const OverlayContext& c, double cx, double cy, double rx, double ry) {
  const OfxPointD p[2] = {{cx - rx, cy - ry}, {cx + rx, cy + ry}};
  c.draw->draw(c.ctx, kOfxDrawPrimitiveEllipse, p, 2);
}

inline void Text(const OverlayContext& c, const std::string& s, double x, double y,
                 int alignment = kOfxDrawTextAlignmentLeft) {
  const OfxPointD p = {x, y};
  c.draw->drawText(c.ctx, s.c_str(), &p, alignment);
}

// A square handle of fixed on-screen size, outlined so it stays visible over
// both bright and dark footage.
inline void Handle(const OverlayContext& c, double x, double y, const Colour& fill,
                   double screenSize = 5.0) {
  const double hx = c.sx(screenSize), hy = c.sy(screenSize);
  SetColour(c, {0.0f, 0.0f, 0.0f, 0.7f});
  FillRect(c, x - hx - c.sx(1), y - hy - c.sy(1), x + hx + c.sx(1), y + hy + c.sy(1));
  SetColour(c, fill);
  FillRect(c, x - hx, y - hy, x + hx, y + hy);
}

inline void Panel(const OverlayContext& c, const OfxRectD& r) {
  SetColour(c, colours::kPanel);
  FillRect(c, r.x1, r.y1, r.x2, r.y2);
  SetColour(c, colours::kPanelEdge);
  SetLineWidth(c, 1.0f);
  StrokeRect(c, r.x1, r.y1, r.x2, r.y2);
}

// A frame number for display: whole frames, rounded half away from zero so
// -0.5 reads as -1 and not as 0.
inline std::string FrameLabel(double frame) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%d", int(frame + (frame < 0.0 ? -0.5 : 0.5)));
  return buf;
}

inline double SnapTo(double v, double step) {
  if (step <= 0.0) return v;
  return std::floor(v / step + 0.5) * step;
}

inline bool Contains(const OfxRectD& r, const OfxPointD& p) {
  return p.x >= r.x1 && p.x <= r.x2 && p.y >= r.y1 && p.y <= r.y2;
}

// Hit test with a fixed on-screen tolerance, so grabbing a handle feels the
// same at any zoom level.
inline bool NearPoint(const OverlayContext& c, const OfxPointD& p, double x, double y,
                      double screenRadius = 8.0) {
  return p.x >= x - c.sx(screenRadius) && p.x <= x + c.sx(screenRadius) &&
         p.y >= y - c.sy(screenRadius) && p.y <= y + c.sy(screenRadius);
}

}  // namespace rta
