#include "overlay.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "ofxDrawSuite.h"
#include "ofxsInteract.h"

#include "clip_time.h"
#include "curve_widget.h"
#include "draw_utils.h"

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
    } catch (...) {
    }
  }

  bool draw(const OFX::DrawArgs& args) override {
    OverlayContext c;
    if (!build(c, args.time, args.pixelScale, gDrawContext)) return false;
    _curve.layout(c);
    _curve.draw(c);
    return true;
  }

  bool penDown(const OFX::PenArgs& args) override {
    OverlayContext c;
    if (!build(c, args.time, args.pixelScale, nullptr)) return false;
    _curve.layout(c);
    return _curve.penDown(c, args.penPosition);
  }

  bool penMotion(const OFX::PenArgs& args) override {
    if (!_curve.dragging()) return false;
    OverlayContext c;
    if (!build(c, args.time, args.pixelScale, nullptr)) return false;
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
      if (!_effect->fetchBooleanParam("showCurveEditor")->getValueAtTime(time)) return false;

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
      a.easing = Easing(std::clamp(getChoice(_effect, "easing"), 0, 4));
      a.bezier.x1 = float(getD(_effect, "easeX1", time));
      a.bezier.y1 = float(getD(_effect, "easeY1", time));
      a.bezier.x2 = float(getD(_effect, "easeX2", time));
      a.bezier.y2 = float(getD(_effect, "easeY2", time));
      a.startTime = getD(_effect, "startTime", time);
      a.duration = getD(_effect, "duration", time);

      // Playhead tracks the first group's progress -- that is the curve the
      // handles are shaping. Uses the same clip-relative origin as the renderer,
      // or the dot would drift away from what is actually on screen.
      const double fps =
          safeFrameRate(src, _effect->fetchClip(kOfxImageEffectOutputClipName));
      const double seconds = toClipTime(_effect, time) / fps;
      out.progress = (seconds - a.startTime) / std::max(1e-6, a.duration);
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
