#include "overlay.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "ofxDrawSuite.h"
#include "ofxsInteract.h"

#include "clip_time.h"
#include "curve_widget.h"
#include "draw_utils.h"
#include "warning_state.h"

namespace rta {
namespace {

// Fetched once, optionally: a host without the draw suite gets no overlay
// rather than a plugin that fails to load.
OfxDrawSuiteV1* drawSuite() {
  static OfxDrawSuiteV1* suite = const_cast<OfxDrawSuiteV1*>(
      static_cast<const OfxDrawSuiteV1*>(OFX::fetchSuite(kOfxDrawSuite, 1, /*optional*/ true)));
  return suite;
}

// The draw context is only valid for the duration of one draw action, and the
// Support library's DrawArgs does not carry it. The shim below stashes it here
// immediately before delegating, and clears it straight after.
OfxDrawContextHandle gDrawContext = nullptr;

double getD(OFX::ImageEffect* e, const char* name, double t) {
  return e->fetchDoubleParam(name)->getValueAtTime(t);
}

int getChoice(OFX::ImageEffect* e, const char* name) {
  int v = 0;
  e->fetchChoiceParam(name)->getValue(v);
  return v;
}

class CurveInteract : public OFX::OverlayInteract {
 public:
  CurveInteract(OfxInteractHandle handle, OFX::ImageEffect* effect)
      : OFX::OverlayInteract(handle), _effect(effect) {
    // Slaving makes the host redraw the overlay when a value changes, so edits
    // made in the Inspector show up immediately rather than on the next
    // viewer refresh. Purely an optimisation -- if it throws, the overlay still
    // works, it just redraws less eagerly.
    if (!_effect) return;
    try {
      addParamToSlaveTo(_effect->fetchChoiceParam("easing"));
      addParamToSlaveTo(_effect->fetchDoubleParam("easeX1"));
      addParamToSlaveTo(_effect->fetchDoubleParam("easeY1"));
      addParamToSlaveTo(_effect->fetchDoubleParam("easeX2"));
      addParamToSlaveTo(_effect->fetchDoubleParam("easeY2"));
      addParamToSlaveTo(_effect->fetchBooleanParam("showCurveEditor"));
      // The slant guides have to follow their own controls, or dragging the
      // angle would leave the lines behind until something else forced a redraw.
      addParamToSlaveTo(_effect->fetchBooleanParam("showDiagnostics"));
      addParamToSlaveTo(_effect->fetchBooleanParam("autoSlant"));
      addParamToSlaveTo(_effect->fetchDoubleParam("italicSlant"));
    } catch (...) {
    }
  }

  bool draw(const OFX::DrawArgs& args) override {
    OverlayContext c;
    if (!build(c, args.time, args.pixelScale, gDrawContext)) return false;

    // Warnings first, and independently of the curve editor: they are the whole
    // reason to look at the overlay when something is wrong.
    drawWarnings(c);
    drawSlant(c);

    if (!showCurve(args.time)) return true;
    _curve.layout(c);
    _curve.draw(c);
    return true;
  }

  // Red, large, top-left of the image. Resolve gives a plugin no colour control
  // in the Inspector, truncates the label and draws a value box beside it -- so
  // that is a poor place for an alert and this is a good one.
  void drawWarnings(const OverlayContext& c) {
    const WarningState w = warningState(_effect);
    if (!w.any()) return;

    const double x = c.rod.x1 + c.sx(24.0);
    double y = c.rod.y2 - c.sy(28.0);

    SetColour(c, Colour{1.0f, 0.25f, 0.25f, 1.0f});
    if (w.clipTooShort) {
      Text(c, "(!) CLIP TOO SHORT - the animation cannot finish. Make the clip longer.", x, y,
           kOfxDrawTextAlignmentLeft | kOfxDrawTextAlignmentTop);
      y -= c.sy(22.0);
    }
    if (w.sourceOffset) {
      Text(c, "(!) SOURCE OFFSET - reveal starts late. Lower Start (frames), or work in Fusion.",
           x, y, kOfxDrawTextAlignmentLeft | kOfxDrawTextAlignmentTop);
    }
  }

  // The italic angle, drawn as lines you can match against the letters. A number
  // in the Inspector says nothing about whether it fits the type; a line lying
  // along the stems says everything, which is the difference between adjusting
  // the angle and guessing at it.
  //
  // Shown with Show Detection, since that is already the detection-tuning mode.
  void drawSlant(const OverlayContext& c) {
    bool show = false;
    try {
      show = _effect->fetchBooleanParam("showDiagnostics")->getValueAtTime(c.time);
    } catch (...) {
      return;
    }
    if (!show) return;

    const WarningState w = warningState(_effect);
    const AnalysisState a = analysisState(_effect);
    if (!a.haveSlant) return;

    bool autoOn = true;
    try {
      autoOn = _effect->fetchBooleanParam("autoSlant")->getValueAtTime(c.time);
    } catch (...) {
    }

    // Positive degrees lean right, and +y is up in canonical space, so the top
    // of each guide moves right by height * tan(angle).
    const double rad = double(a.slantDegrees) * 3.14159265358979323846 / 180.0;
    const double h = c.rod.y2 - c.rod.y1;
    const double dx = std::tan(rad) * h;

    SetColour(c, Colour{0.20f, 0.85f, 1.0f, 0.55f});
    SetLineWidth(c, 1.0f);
    const int kLines = 9;
    for (int i = 1; i < kLines; ++i) {
      const double f = double(i) / double(kLines);
      const double x = c.rod.x1 + f * (c.rod.x2 - c.rod.x1);
      Line(c, x - dx * 0.5, c.rod.y1, x + dx * 0.5, c.rod.y2);
    }

    char buf[96];
    std::snprintf(buf, sizeof(buf), "ITALIC %.1f deg  %s", a.slantDegrees,
                  autoOn ? "(auto)" : "(manual)");
    SetColour(c, Colour{0.20f, 0.85f, 1.0f, 1.0f});
    // Below any warning text, so the two never sit on top of each other.
    const double y = c.rod.y2 - c.sy(w.any() ? 72.0 : 28.0);
    Text(c, buf, c.rod.x1 + c.sx(24.0), y,
         kOfxDrawTextAlignmentLeft | kOfxDrawTextAlignmentTop);
  }

  bool showCurve(double time) {
    try {
      return _effect->fetchBooleanParam("showCurveEditor")->getValueAtTime(time);
    } catch (...) {
      return false;
    }
  }

  bool penDown(const OFX::PenArgs& args) override {
    OverlayContext c;
    if (!build(c, args.time, args.pixelScale, nullptr) || !showCurve(args.time)) return false;
    _curve.layout(c);
    return _curve.penDown(c, args.penPosition);
  }

  bool penMotion(const OFX::PenArgs& args) override {
    if (!_curve.dragging()) return false;
    OverlayContext c;
    if (!build(c, args.time, args.pixelScale, nullptr) || !showCurve(args.time)) return false;
    _curve.layout(c);
    return _curve.penMotion(c, args.penPosition);
  }

  bool penUp(const OFX::PenArgs& args) override {
    OverlayContext c;
    if (!build(c, args.time, args.pixelScale, nullptr)) return false;
    return _curve.penUp(c, args.penPosition);
  }

  void loseFocus(const OFX::FocusArgs& args) override {
    // A drag released outside the viewer, or interrupted by a page switch,
    // never delivers its mouse-up. Without this the edit block it opened stays
    // open for the rest of the session and the host's parameter state is left
    // inconsistent.
    if (!_curve.dragging()) return;
    OverlayContext c;
    if (build(c, args.time, args.pixelScale, nullptr)) _curve.abandon(c);
  }

 private:
  // Snapshots parameters and viewport for one event. Never throws: an exception
  // escaping into the host mid-draw is far worse than a frame without overlay.
  bool build(OverlayContext& out, double time, const OfxPointD& pixelScale,
             OfxDrawContextHandle ctx) {
    try {
      if (!_effect || !drawSuite()) return false;

      OFX::Clip* src = _effect->fetchClip(kOfxImageEffectSimpleSourceClipName);
      if (!src || !src->isConnected()) return false;

      out.effect = _effect;
      out.draw = drawSuite();
      out.ctx = ctx;
      out.time = time;
      out.pixelScale = pixelScale;
      out.rod = src->getRegionOfDefinition(time);

      if (out.rod.x2 - out.rod.x1 <= 0.0 || out.rod.y2 - out.rod.y1 <= 0.0) return false;
      // A degenerate pixel scale collapses every screen-relative size to zero
      // and the overlay silently vanishes.
      if (out.pixelScale.x <= 0.0) out.pixelScale.x = 1.0;
      if (out.pixelScale.y <= 0.0) out.pixelScale.y = 1.0;

      AnimParams& a = out.anim;
      a.in.easing = Easing(std::clamp(getChoice(_effect, "easing"), 0, 4));
      a.in.bezier.x1 = float(getD(_effect, "easeX1", time));
      a.in.bezier.y1 = float(getD(_effect, "easeY1", time));
      a.in.bezier.x2 = float(getD(_effect, "easeX2", time));
      a.in.bezier.y2 = float(getD(_effect, "easeY2", time));
      a.startTime = getD(_effect, "startTime", time);
      a.in.duration = getD(_effect, "duration", time);

      // Playhead tracks the first group's progress -- that is the curve the
      // handles are shaping. Same clip-relative frame origin as the renderer,
      // or the dot would drift away from what is actually on screen.
      const double frames = toClipTime(_effect, time);
      out.progress = (frames - a.startTime) / std::max(1e-6, a.in.duration);
      out.hasProgress = out.progress >= 0.0 && out.progress <= 1.0;
      return true;
    } catch (...) {
      return false;
    }
  }

  OFX::ImageEffect* _effect = nullptr;
  CurveWidget _curve;
};

// Wraps the Support library's interact entry so the draw context, which only
// ever appears in the draw action's inArgs, can be captured on the way past.
// Everything else -- instance lifetime, pen dispatch, param slaving -- is left
// to the Support library.
class CurveOverlayDescriptor
    : public OFX::DefaultEffectOverlayDescriptor<CurveOverlayDescriptor, CurveInteract> {
 public:
  OfxPluginEntryPoint* getMainEntry() override { return &entry; }

 private:
  static OfxStatus entry(const char* action, const void* handle, OfxPropertySetHandle inArgs,
                         OfxPropertySetHandle outArgs) {
    const bool isDraw = std::strcmp(action, kOfxInteractActionDraw) == 0;
    if (isDraw && inArgs) {
      OFX::PropertySet p(inArgs);
      gDrawContext = static_cast<OfxDrawContextHandle>(
          p.propGetPointer(kOfxInteractPropDrawContext, 0, /*throwOnFailure*/ false));
    }

    const OfxStatus st = CurveOverlayDescriptor::overlayInteractMainEntry(action, handle, inArgs,
                                                                          outArgs);
    // The handle is only valid inside the draw action; holding it past that
    // would hand the widget a dangling context on the next pen event.
    if (isDraw) gDrawContext = nullptr;
    return st;
  }
};

}  // namespace

void registerOverlay(OFX::ImageEffectDescriptor& desc) {
  desc.setOverlayInteractDescriptor(new CurveOverlayDescriptor());
}

}  // namespace rta
