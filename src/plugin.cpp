// OFX host glue. Everything interesting lives in segmentation/animator/
// compositor; this file only fetches images, reads parameters, and maps OFX
// coordinates into the flat local space the core code works in.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <atomic>
#include <mutex>
#include <vector>

#include "ofxsImageEffect.h"
#include "ofxsMultiThread.h"

#include "animator.h"
#include "clip_time.h"
#include "compositor.h"
#include "edit_block.h"
#include "cuda_compositor.h"
#include "diagnostics.h"
#include "interact/overlay.h"
#include "probe.h"
#include "segmentation.h"
#include "warning_state.h"

#define kPluginName "Text Animator"
#define kPluginGrouping "Text"
#define kPluginIdentifier "com.johnlwi.TextAnimator"
#define kPluginDescription                                                         \
  "Reveals rendered text one character, word or line at a time by detecting "      \
  "glyphs directly from the alpha channel. Apply to a title clip on its own "      \
  "track; no text metadata is required."

namespace {

// ChoiceParam is the one param type without a return-value getter.
int choiceAt(OFX::ChoiceParam* p, double t, int lo, int hi) {
  int v = 0;
  p->getValueAtTime(t, v);
  return std::clamp(v, lo, hi);
}

// Single funnel for adding parameters to the page.
//
// Every parameter declares that it invalidates the whole cache, which is what
// MultiTransform does and for the same measured reason: Resolve keeps a
// per-frame verdict about an effect and will happily keep serving frames it
// rendered earlier. Its notes describe the symptom precisely -- "edits applied
// while the playhead sat outside the range appeared to do nothing at all, and
// only a manual cache purge fixed it".
//
// The OFX default, eCacheInvalidateValueChange, means "invalidate only the
// frames this parameter's keyframe covers". That is right for a keyframed
// effect and wrong here: nothing in this plugin is ever keyframed, the
// parameters are constants the animator interprets against the render time, so
// every parameter affects every frame.
//
// Every parameter here declares that it invalidates the whole cache.
//
// Nothing in this plugin is ever keyframed: the parameters are constants the
// animator interprets against the render time, so *every* parameter affects
// *every* frame. The OFX default, eCacheInvalidateValueChange, only invalidates
// the range a parameter's own keyframe covers, which gives a caching host no
// reason to discard anything -- so it keeps replaying frames rendered before
// the edit. Resolve tolerates that; Fusion does not, and Fusion is right.
//
// This was removed once after project loading broke, but that was never
// isolated -- it was eliminated by assumption while bisecting four changes at
// once, and MultiTransform ships it without trouble. If loading breaks again,
// this is the first suspect and the next step is narrowing it to the handful of
// parameters that genuinely need it.
template <class T>
void addParam(OFX::PageParamDescriptor* page, T* p) {
  p->setCacheInvalidation(OFX::eCacheInvalidateValueAll);
  page->addChild(*p);
}

// For parameters that change nothing about the rendered image -- the overlay
// toggle, the status line. Purging every cached frame because someone opened
// the curve editor would be a stall for no reason.
template <class T>
void addParamUI(OFX::PageParamDescriptor* page, T* p) {
  page->addChild(*p);
}

// OFX images are bottom-up: y=0 is the bottom row. The core code is top-down
// (line 0 is the top line, +y slides downward). Converting here, at the single
// boundary, is what keeps "animate the first line first" and "90 degrees rises
// from below" meaning the same thing in Resolve as in the test harness.
rta::RectI toLocal(const OfxRectI& r, const OfxRectI& region) {
  return rta::RectI{r.x1 - region.x1, region.y2 - r.y2, r.x2 - region.x1, region.y2 - r.y1};
}

// Builds a top-down view whose (0,0) is the top-left of `region`, so src and
// dst share one coordinate space even when the host hands back differently
// placed buffers. The flip is a negative row stride, so no pixels are moved.
rta::ImageView makeView(OFX::Image* img, const OfxRectI& region) {
  rta::ImageView v;
  if (!img) return v;
  void* base = img->getPixelAddress(region.x1, region.y2 - 1);  // topmost row
  if (!base) return v;
  v.data = static_cast<float*>(base);
  v.width = region.x2 - region.x1;
  v.height = region.y2 - region.y1;
  v.rowStride = -std::ptrdiff_t(img->getRowBytes()) / std::ptrdiff_t(sizeof(float));
  return v;
}

// Re-labelling every frame would be wasteful while scrubbing a static title,
// so keep the last result and reuse it when the input and settings match.
// Held by shared_ptr so a cache hit costs a refcount bump. Copying the whole
// Segmentation per frame would mean memcpy'ing the label image -- 8 MB at
// 1080p, 33 MB at UHD -- on every single render, while holding the mutex.
struct AnalysisCache {
  std::mutex mutex;
  uint64_t hash = 0;
  int width = 0, height = 0;
  rta::DetectParams base;  // detection settings other than the grouping mode

  // One segmentation per grouping mode, because the entrance and the exit may
  // group differently -- words in, characters out. At most two are ever filled.
  std::shared_ptr<const rta::Segmentation> byMode[3];

  void invalidate() {
    for (auto& s : byMode) s.reset();
#if RTA_WITH_CUDA
    devSegMode = -1;
    devSegValid = false;
#endif
  }

#if RTA_WITH_CUDA
  // Device copy of whichever segmentation the current frame actually uses.
  rta::CudaSegmentation devSeg;
  rta::CudaScratch scratch;
  int devSegMode = -1;
  bool devSegValid = false;
#endif
};

// Detection settings compare equal ignoring the grouping mode, which is what
// decides whether the cached label image can be reused across modes.
bool sameBase(const rta::DetectParams& a, const rta::DetectParams& b) {
  rta::DetectParams x = a, y = b;
  x.mode = y.mode = rta::GroupMode::Word;
  return x == y;
}

class TextAnimatorPlugin : public OFX::ImageEffect {
 public:
  explicit TextAnimatorPlugin(OfxImageEffectHandle handle) : OFX::ImageEffect(handle) {
    _dstClip = fetchClip(kOfxImageEffectOutputClipName);
    _srcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);

    _groupMode = fetchChoiceParam("groupMode");
    _animation = fetchChoiceParam("animation");
    _easing = fetchChoiceParam("easing");
    _order = fetchChoiceParam("order");
    _lineOrder = fetchChoiceParam("lineOrder");
    _randomSeed = fetchIntParam("randomSeed");
    _startTime = fetchDoubleParam("startTime");
    _duration = fetchDoubleParam("duration");
    _stagger = fetchDoubleParam("stagger");
    _slideDistance = fetchDoubleParam("slideDistance");
    _slideAngle = fetchDoubleParam("slideAngle");
    _startScale = fetchDoubleParam("startScale");
    _startRotation = fetchDoubleParam("startRotation");
    _startBlur = fetchDoubleParam("startBlur");
    _motionBlur = fetchBooleanParam("motionBlur");
    _shutterAngle = fetchDoubleParam("shutterAngle");
    _blurSamples = fetchIntParam("blurSamples");
    _distanceUnits = fetchChoiceParam("distanceUnits");

    _enableIn = fetchBooleanParam("enableIn");
    _enableOut = fetchBooleanParam("enableOut");
    _outOffset = fetchDoubleParam("outOffset");
    _linkOut = fetchBooleanParam("linkOut");
    _mirrorOut = fetchBooleanParam("mirrorOut");
    _clipLengthOverride = fetchDoubleParam("clipLengthOverride");
    _manualOrigin = fetchBooleanParam("manualOrigin");
    _originFrame = fetchDoubleParam("originFrame");

    _outGroupMode = fetchChoiceParam("outGroupMode");
    _outAnimation = fetchChoiceParam("outAnimation");
    _outEasing = fetchChoiceParam("outEasing");
    _outOrder = fetchChoiceParam("outOrder");
    _outLineOrder = fetchChoiceParam("outLineOrder");
    _outRandomSeed = fetchIntParam("outRandomSeed");
    _outDuration = fetchDoubleParam("outDuration");
    _outStagger = fetchDoubleParam("outStagger");
    _outSlideDistance = fetchDoubleParam("outSlideDistance");
    _outSlideAngle = fetchDoubleParam("outSlideAngle");
    _outStartScale = fetchDoubleParam("outStartScale");
    _outStartRotation = fetchDoubleParam("outStartRotation");
    _outStartBlur = fetchDoubleParam("outStartBlur");
    _easeX1 = fetchDoubleParam("easeX1");
    _easeY1 = fetchDoubleParam("easeY1");
    _easeX2 = fetchDoubleParam("easeX2");
    _easeY2 = fetchDoubleParam("easeY2");

    _alphaThreshold = fetchDoubleParam("alphaThreshold");
    _minBlobArea = fetchIntParam("minBlobArea");
    _wordGapSensitivity = fetchDoubleParam("wordGapSensitivity");
    _bridgeRadius = fetchIntParam("bridgeRadius");
    _showDiagnostics = fetchBooleanParam("showDiagnostics");
    _hostWarning = fetchStringParam("hostWarning");
    updateHostWarning();
  }

  ~TextAnimatorPlugin() override { rta::clearWarningState(this); }

  void render(const OFX::RenderArguments& args) override;

  bool isIdentity(const OFX::IsIdentityArguments&, OFX::Clip*&, double&) override {
    // Never a pass-through, at any time.
    //
    // Answering per frame is the trap MultiTransform documents: outside its own
    // animation an effect looks like the identity, so it declares itself a
    // pass-through on exactly those frames -- and the host caches that verdict
    // per frame. Move the animation to cover one of them and the host stays
    // convinced nothing happens there, so the edit appears to do nothing until
    // the cache is purged by hand.
    //
    // This plugin is never a pass-through anyway: even fully settled it has
    // recomposited the title from its own segmentation. Saying so at every time
    // leaves no per-frame verdict to go stale.
    return false;
  }

  // The frame that counts as clip time 0. Used by both render and the capture
  // buttons -- and it is essential that it is the SAME function for both.
  //
  // Resolve reports no usable clip start (every route describes the available
  // media, so it moves with the head handle), and that turns out not to matter:
  // if a button captures the playhead through this same mapping, whatever error
  // is in the origin cancels out against the identical error at render time.
  // The animation then begins exactly where the playhead was parked, however
  // the clip is trimmed. This is how MultiTransform sidesteps the same gap.
  double originAt(double) {
    // The clip BOUNDARY is frame 0, and here it already is.
    //
    // MultiTransform subtracts t1 from the render time because its host hands
    // out absolute timeline frames (~108000), so the subtraction is what maps
    // the boundary to zero. Measured here, Resolve hands this plugin
    // clip-relative time already: args.time never goes below 0 and never
    // negative, whatever the Inspector labels the frame. The boundary is
    // therefore already 0 and the correct offset is none.
    //
    // Subtracting anything -- t1, an anchor parameter, a learned value -- moves
    // the animation off the clip's first frame by exactly that amount. t1 was
    // the worst of them: eleven different values on one clip, sliding across
    // the range its head had been extended by.
    //
    // t1 straight from the host, with no state of any kind.
    //
    // This is deliberately the version that trims correctly. t1 moves with the
    // clip, so shortening it -- or moving it along the timeline -- re-anchors
    // the animation immediately. Its one failure is extending the head, where
    // t1 starts including the unused handle and the reveal begins early; that
    // residue is what Start (frames) offsets and what the timing indicator
    // flags.
    //
    // The alternative, a running minimum of render times, has the mirror-image
    // fault and a worse one: it follows an extension but can never follow a
    // shorter trim, because it only ever falls. And every attempt to let it
    // re-learn -- on bounds change, on t2 change -- reset it to the frame being
    // rendered, pinning clip time to 0 so the entrance never started at all.
    // Three separate versions broke that way. Being wrong only when the head is
    // extended is a much smaller fault than not animating.
    double start = 0.0, length = 0.0;
    if (rta::getClipRange(this, start, length)) return start;
    return 0.0;
  }


  // Feed a render time into the running minimum. Must happen before originAt()
  // is consulted for that same frame, or the clip's first frame would be
  // measured against an origin that does not yet include it.
  void observeRenderTime(double time) {
    double rawT1 = 0.0, rawT2 = 0.0;
    bool haveBounds = true;
    try {
      timeLineGetBounds(rawT1, rawT2);
    } catch (...) {
      haveBounds = false;
    }

    std::lock_guard<std::mutex> lock(_seenMutex);

    // Deliberately NOT reset whenever the reported bounds change. Those values
    // jitter constantly while a clip is dragged -- dozens of distinct values in
    // a single session -- and resetting on each one pinned the origin to the
    // frame being rendered, making clip time permanently 0. Frame 0 of an
    // entrance is zero opacity, so the text simply sat invisible until the
    // bounds happened to settle.
    //
    // Monotonically decreasing. Nothing here ever raises it.
    //
    // Re-learning on a bounds change was tried and reverted: it set the anchor
    // to the frame being rendered, and the bounds do change -- t2 was measured
    // flipping between 0 and 119 on one clip, t1 across eleven values. Each
    // flip pinned clip time back to 0, which is an entrance's first frame, so
    // nothing was drawn and the reveal never started. That was the intended fix
    // for a head-trim offset and it cost far more than it bought.
    //
    // A value that can only fall cannot move the animation backwards, and the
    // lowest frame the host ever asks for is the clip's first frame. The
    // residual head-trim offset on Fusion clips is handled by Start (frames),
    // and flagged by the timing indicator rather than guessed at.
    if (time < _seenEarliest) _seenEarliest = time;
    _seenT1 = rawT1;
    _seenT2 = rawT2;
  }

  void changedParam(const OFX::InstanceChangedArgs&, const std::string&) override {
    // Nothing to capture. The clip boundary is already frame 0, so there is no
    // anchor to set -- and the buttons that used to write one are what pinned a
    // -10 into the clip and shifted every animation by the length of the head
    // extension. The controls are gone from the UI; the parameters remain
    // defined so existing projects still load, and are ignored.
  }

  // Warn when the host's clip bounds say the timing cannot be trusted.
  //
  // A negative t1 means the clip reaches back before its source start -- head
  // handle on a Fusion clip, which is the case where Resolve's render time
  // follows the composition rather than the trimmed clip and the reveal ends up
  // offset. A PNG or a plain still never reports this, and those work
  // correctly, so the sign of t1 separates the two cleanly.
  //
  // Written to the LABEL, not the value: Resolve draws only the label of a
  // string parameter, so a message left in the value would be invisible.
  void updateHostWarning() {
    double t1 = 0.0, t2 = 0.0;
    bool ok = true;
    try {
      timeLineGetBounds(t1, t2);
    } catch (...) {
      ok = false;
    }
    // Short. Resolve truncates a parameter label to roughly the left half of
    // the row -- the rest of the width goes to the value field -- so a sentence
    // here just renders as "(!) Source ...usion comp.". The explanation lives
    // in the tooltip, which has room.
    // Clip-too-short wins: it is the one the user can act on directly.
    const char* text = "Timing OK";
    if (!_fitOk.load(std::memory_order_relaxed))
      text = "(!) Clip too short";
    else if (ok && t1 < 0.0)
      text = "(!) Source offset";

    try {
      _hostWarning->setLabel(text);
      // The field itself is meaningless; greying it out stops it reading as
      // something to type into.
      _hostWarning->setEnabled(false);
    } catch (...) {
    }
  }

  void changedClip(const OFX::InstanceChangedArgs&, const std::string&) override {
    updateHostWarning();
  }

  void getClipPreferences(OFX::ClipPreferencesSetter& prefs) override {
    // Clip prefs are recomputed when the clip or its trim changes, which is
    // exactly when this needs re-checking.
    updateHostWarning();
    // Declare that the output changes over time even though no parameter is
    // animated.
    //
    // The animation is internal: driven by the render time, not by host
    // keyframes. Without this a host is entirely within its rights to render
    // one frame and reuse it for the whole clip, because as far as it can tell
    // nothing about the effect varies. Fusion does exactly that -- playback
    // shows a frozen image while the controls still update it live -- whereas
    // Resolve's Edit page happens to re-render regardless.
    prefs.setOutputFrameVarying(true);
  }

 private:
  OFX::Clip* _dstClip = nullptr;
  OFX::Clip* _srcClip = nullptr;

  OFX::ChoiceParam *_groupMode, *_animation, *_easing, *_order, *_lineOrder, *_distanceUnits;
  OFX::IntParam *_randomSeed, *_minBlobArea, *_bridgeRadius, *_blurSamples;
  OFX::DoubleParam *_startTime, *_duration, *_stagger, *_slideDistance, *_slideAngle, *_startScale;
  OFX::DoubleParam *_startRotation, *_startBlur, *_shutterAngle;
  OFX::DoubleParam *_easeX1, *_easeY1, *_easeX2, *_easeY2;
  OFX::DoubleParam *_alphaThreshold, *_wordGapSensitivity;
  OFX::BooleanParam *_showDiagnostics, *_motionBlur;
  OFX::StringParam* _hostWarning;

  // Exit stage.
  OFX::BooleanParam *_enableIn, *_enableOut, *_linkOut, *_mirrorOut;
  OFX::DoubleParam *_outOffset, *_clipLengthOverride, *_originFrame;
  OFX::BooleanParam* _manualOrigin;
  OFX::ChoiceParam *_outGroupMode, *_outAnimation, *_outEasing, *_outOrder, *_outLineOrder;
  OFX::IntParam* _outRandomSeed;
  OFX::DoubleParam *_outDuration, *_outStagger, *_outSlideDistance, *_outSlideAngle;
  OFX::DoubleParam *_outStartScale, *_outStartRotation, *_outStartBlur;

  AnalysisCache _cache;

  // Lowest render time seen since the clip's reported bounds last changed --
  // i.e. the clip's first frame, which is the only place it can be learned.
  // Reset on a bounds change so re-trimming re-learns it.
  // Set by render, read by the indicator: does every animation fit the clip?
  std::atomic<bool> _fitOk{true};

  std::mutex _seenMutex;
  double _seenT1 = 1e300, _seenT2 = 1e300, _seenEarliest = 1e300;

  // Carried between the two halves of the diagnostic block only.
  double _probeRaw[2] = {0.0, 0.0};
  bool _probeRawOk = false;
};

void TextAnimatorPlugin::render(const OFX::RenderArguments& args) {
  // FIRST STATEMENT, deliberately. The auto origin is the lowest frame the host
  // has asked for, and every one of the early returns below -- an unfetchable
  // image, an unsupported depth, a failed CUDA call, an empty segmentation --
  // would otherwise hide that frame from the running minimum. Miss the clip's
  // first frames that way and the origin stays at whatever it learned earlier,
  // which lands the animation at the clip's *previous* start.
  observeRenderTime(args.time);

  // Trace every frame and how it left.
  //
  // The other probes sit deep in render(), after a dozen early returns, so a
  // frame that bailed never appeared in them at all -- the log looked clean
  // precisely because the broken frames were the ones missing from it. This
  // logs from a destructor, so it reports no matter which return is taken.
  struct Trace {
    double t;
    const char* why = "BAILED-early";
    ~Trace() {
      char b[64];
      std::snprintf(b, sizeof(b), "t=%.1f %s", t, why);
      rta::probeOnChange("render", b);
    }
  } trace{args.time};

  std::unique_ptr<OFX::Image> dst(_dstClip->fetchImage(args.time));
  if (!dst.get()) {
    trace.why = "no-dst-image";
    return;
  }

  if (dst->getPixelDepth() != OFX::eBitDepthFloat ||
      dst->getPixelComponents() != OFX::ePixelComponentRGBA) {
    trace.why = "bad-dst-format";
    OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    return;
  }

  std::unique_ptr<OFX::Image> src(
      (_srcClip && _srcClip->isConnected()) ? _srcClip->fetchImage(args.time) : nullptr);

  const OfxRectI dstBounds = dst->getBounds();

  // When Resolve renders on the GPU the pixel pointers are CUDA device memory,
  // so nothing below may dereference them on the host.
  bool useCuda = false;
#if RTA_WITH_CUDA
  useCuda = args.isEnabledCudaRender && rta::cudaAvailable();
#endif

  // Clear the whole render window first: anything we don't draw is transparent.
  {
    const rta::ImageView full = makeView(dst.get(), dstBounds);
    if (full.valid()) {
      const rta::RectI w = toLocal(args.renderWindow, dstBounds);
      const rta::RectI clip{std::max(0, w.x1), std::max(0, w.y1), std::min(full.width, w.x2),
                            std::min(full.height, w.y2)};
#if RTA_WITH_CUDA
      if (useCuda) {
        rta::cudaClearWindow(full.data, full.rowStride, full.width, full.height, clip,
                             args.pCudaStream);
      } else
#endif
      {
        for (int y = clip.y1; y < clip.y2; ++y) {
          if (clip.width() > 0)
            std::fill(full.at(clip.x1, y), full.at(clip.x1, y) + size_t(clip.width()) * 4, 0.0f);
        }
      }
    }
  }

  if (!src.get() || src->getPixelDepth() != OFX::eBitDepthFloat ||
      src->getPixelComponents() != OFX::ePixelComponentRGBA) {
    trace.why = "no-src-image";
    return;
  }

  // Work in the region both images actually cover.
  const OfxRectI srcBounds = src->getBounds();
  OfxRectI common{std::max(srcBounds.x1, dstBounds.x1), std::max(srcBounds.y1, dstBounds.y1),
                  std::min(srcBounds.x2, dstBounds.x2), std::min(srcBounds.y2, dstBounds.y2)};
  if (common.x2 <= common.x1 || common.y2 <= common.y1) {
    trace.why = "empty-common-region";
    return;
  }

  const rta::ImageView srcView = makeView(src.get(), common);
  const rta::ImageView dstView = makeView(dst.get(), common);
  if (!srcView.valid() || !dstView.valid()) {
    trace.why = "invalid-view";
    return;
  }

  // ------------------------------------------------------------ parameters
  rta::DetectParams det;
  det.alphaThreshold = float(_alphaThreshold->getValueAtTime(args.time));
  det.minBlobArea = std::max(1, _minBlobArea->getValueAtTime(args.time));
  det.wordGapSensitivity = float(_wordGapSensitivity->getValueAtTime(args.time));
  det.bridgeRadius = std::max(0, _bridgeRadius->getValueAtTime(args.time));
  det.mode = rta::GroupMode(choiceAt(_groupMode, args.time, 0, 2));

  rta::AnimParams anim;
  anim.enableIn = _enableIn->getValueAtTime(args.time);
  anim.enableOut = _enableOut->getValueAtTime(args.time);
  anim.startTime = _startTime->getValueAtTime(args.time);
  anim.outOffset = _outOffset->getValueAtTime(args.time);
  anim.motionBlur = _motionBlur->getValueAtTime(args.time);
  anim.shutterAngle = _shutterAngle->getValueAtTime(args.time);
  anim.blurSamples = _blurSamples->getValueAtTime(args.time);

  rta::StageSettings& in = anim.in;
  in.animation = rta::Animation(choiceAt(_animation, args.time, 0, 1));
  in.easing = rta::Easing(choiceAt(_easing, args.time, 0, 4));
  in.bezier.x1 = float(_easeX1->getValueAtTime(args.time));
  in.bezier.y1 = float(_easeY1->getValueAtTime(args.time));
  in.bezier.x2 = float(_easeX2->getValueAtTime(args.time));
  in.bezier.y2 = float(_easeY2->getValueAtTime(args.time));
  in.order = rta::Order(choiceAt(_order, args.time, 0, 3));
  in.lineOrder = rta::LineOrder(choiceAt(_lineOrder, args.time, 0, 1));
  in.randomSeed = _randomSeed->getValueAtTime(args.time);
  in.duration = _duration->getValueAtTime(args.time);
  in.stagger = _stagger->getValueAtTime(args.time);
  in.slideAngle = _slideAngle->getValueAtTime(args.time);
  in.startScale = _startScale->getValueAtTime(args.time);
  in.startRotation = _startRotation->getValueAtTime(args.time);

  // Slide distance and blur are lengths, so they are converted below once the
  // segmentation is known -- "% of text height" needs the measured glyphs.
  const int distanceUnits = choiceAt(_distanceUnits, args.time, 0, 2);
  const double rawSlideIn = _slideDistance->getValueAtTime(args.time);
  const double rawBlurIn = _startBlur->getValueAtTime(args.time);

  // The exit either copies the entrance -- optionally mirrored -- or stands on
  // its own set of controls.
  const bool linkOut = _linkOut->getValueAtTime(args.time);
  const bool mirrorOut = _mirrorOut->getValueAtTime(args.time);
  rta::GroupMode outMode = det.mode;
  double rawSlideOut = rawSlideIn, rawBlurOut = rawBlurIn;

  if (linkOut) {
    anim.out = rta::mirrorStage(in, mirrorOut);
  } else {
    rta::StageSettings& o = anim.out;
    outMode = rta::GroupMode(choiceAt(_outGroupMode, args.time, 0, 2));
    o.animation = rta::Animation(choiceAt(_outAnimation, args.time, 0, 1));
    o.easing = rta::Easing(choiceAt(_outEasing, args.time, 0, 4));
    o.bezier = in.bezier;  // the curve editor shapes a single custom curve
    o.order = rta::Order(choiceAt(_outOrder, args.time, 0, 3));
    o.lineOrder = rta::LineOrder(choiceAt(_outLineOrder, args.time, 0, 1));
    o.randomSeed = _outRandomSeed->getValueAtTime(args.time);
    o.duration = _outDuration->getValueAtTime(args.time);
    o.stagger = _outStagger->getValueAtTime(args.time);
    o.slideAngle = _outSlideAngle->getValueAtTime(args.time);
    o.startScale = _outStartScale->getValueAtTime(args.time);
    o.startRotation = _outStartRotation->getValueAtTime(args.time);
    rawSlideOut = _outSlideDistance->getValueAtTime(args.time);
    rawBlurOut = _outStartBlur->getValueAtTime(args.time);
  }

  // --------------------------------------------------------------- analysis
  uint64_t hash = 0;
#if RTA_WITH_CUDA
  if (useCuda) {
    bool ok = false;
    std::lock_guard<std::mutex> lock(_cache.mutex);
    hash = rta::cudaHashAlpha(srcView.data, srcView.rowStride, srcView.width, srcView.height,
                              det.alphaThreshold, _cache.scratch, args.pCudaStream, &ok);
    if (!ok) {
      trace.why = "cuda-hash-failed";
      return;
    }
  } else
#endif
  {
    hash = rta::hashAlpha(srcView, det.alphaThreshold);
  }

  std::shared_ptr<const rta::Segmentation> segIn, segOut;
  {
    std::lock_guard<std::mutex> lock(_cache.mutex);
    if (_cache.hash != hash || _cache.width != srcView.width ||
        _cache.height != srcView.height || !sameBase(_cache.base, det)) {
      _cache.invalidate();
      _cache.hash = hash;
      _cache.width = srcView.width;
      _cache.height = srcView.height;
      _cache.base = det;
    }

    const bool needIn = !_cache.byMode[int(det.mode)];
    const bool needOut = anim.enableOut && !_cache.byMode[int(outMode)];

    if (needIn || needOut) {
      // One readback serves both modes. This is the only full-frame transfer in
      // the plugin, and it happens per title change rather than per frame.
      std::vector<float> host;
      rta::ImageView hostView = srcView;
#if RTA_WITH_CUDA
      if (useCuda) {
        if (!rta::cudaDownload(host, srcView.data, srcView.rowStride, srcView.width,
                               srcView.height)) {
          trace.why = "cuda-download-failed";
          return;
        }
        const std::ptrdiff_t absStride =
            srcView.rowStride < 0 ? -srcView.rowStride : srcView.rowStride;
        hostView = rta::ImageView{host.data() + absStride * (srcView.height - 1), srcView.width,
                                  srcView.height, -absStride};
      }
#endif
      auto build = [&](rta::GroupMode m) {
        rta::DetectParams d = det;
        d.mode = m;
        _cache.byMode[int(m)] = std::make_shared<const rta::Segmentation>(segment(hostView, d));
      };
      if (needIn) build(det.mode);
      if (needOut) build(outMode);
    }

    segIn = _cache.byMode[int(det.mode)];
    if (anim.enableOut) segOut = _cache.byMode[int(outMode)];
  }
  if (!segIn || segIn->empty()) {
    trace.why = "segmentation-empty";
    return;
  }

  // ---------------------------------------------------------------- animate
  // Clip-relative frames. Trimming the clip's head re-anchors the animation to
  // the new first frame instead of stranding it at a fixed timeline position.
  //
  // No frame rate is involved: timing is authored in frames, which is why this
  // never has to ask the host for kOfxImageEffectPropFrameRate -- a property
  // Fusion does not publish, and which threw out of render when asked for.
  // getFrameRange() is likewise avoided: Resolve returns a 1000-minute sentinel
  // from it rather than the clip's extent. See clip_time.h.
  // Where animation frame 0 sits on the host's timeline.
  //
  // Preferred source is the manual anchor, because Resolve reports no usable
  // clip start: getFrameRange is a 1000-minute sentinel, getUnmappedFrameRange
  // is empty, getEffectDuration returns the available-media span, and
  // timeLineGetBounds t1 is the start minus the head handle. Measured on a clip
  // visibly running 1000..1149: t1 came back 967 and duration 184. Only the
  // playhead knows, so the button captures it.
  const double origin = originAt(args.time);
  const double frames = args.time - origin;

  {
    // Log the RAW bounds, not the clamped ones, or the clamp hides the very
    // thing being diagnosed. `earliest` is the lowest render time seen since
    // the bounds last changed: that, not anything the host reports, is the
    // clip's true first frame, and it is what the origin has to match.
    double rawT1 = 0.0, rawT2 = 0.0;
    bool rawOk = true;
    try {
      timeLineGetBounds(rawT1, rawT2);
    } catch (...) {
      rawOk = false;
    }

    _probeRaw[0] = rawT1;
    _probeRaw[1] = rawT2;
    _probeRawOk = rawOk;
  }

  // The exit is anchored to the clip's END, so it needs the clip's length.
  // Length is measured from the anchor to that end: t2 is the one bound that
  // held still across trims. The override wins when the host reports nothing
  // usable; without either, the exit is skipped rather than guessed at.
  double clipLength = 0.0;
  bool haveLength = false;
  double b1 = 0.0, b2 = 0.0;
  if (rta::getClipBounds(this, b1, b2) && b2 > origin) {
    clipLength = b2 - origin;
    haveLength = true;
  }
  const double lengthOverride = _clipLengthOverride->getValueAtTime(args.time);
  if (lengthOverride > 0.0) {
    clipLength = lengthOverride;
    haveLength = true;
  }

  {
    char buf[380];
    std::snprintf(buf, sizeof(buf),
                  "raw=[%.1f,%.1f]%s dur=%.1f anchor=%s origin=%.1f clipLen=%.1f "
                  "earliestRendered=%.1f | args.time=%.1f -> clipFrame=%.1f",
                  _probeRaw[0], _probeRaw[1], _probeRawOk ? "" : "(THREW)",
                  rta::safeEffectDuration(this),
                  _manualOrigin->getValueAtTime(args.time) ? "manual" : "auto", origin, clipLength,
                  _seenEarliest, args.time, frames);
    rta::probeOnChange("cliptime", buf);
  }

  // The entrance and exit are evaluated TOGETHER when they share a grouping, so
  // an exit that begins before the entrance has settled blends with it instead
  // of cutting it off. With different group modes the two segmentations have
  // different groups and there is nothing to combine, so that case still picks
  // one stage per frame.
  const bool combined = anim.enableOut && outMode == det.mode && segOut && !segOut->empty();

  rta::Stage stage = rta::Stage::Settled;
  const rta::Segmentation* useSeg = segIn.get();
  const rta::StageSettings* useSet = &anim.in;
  rta::GroupMode useMode = det.mode;
  double stageStart = anim.startTime;
  double rawSlide = rawSlideIn, rawBlur = rawBlurIn;

  double outSeqStart = 0.0;
  bool outUsable = false;
  if (anim.enableOut && haveLength && segOut && !segOut->empty()) {
    outSeqStart = clipLength - anim.outOffset - rta::stageSpan(segOut->groups.size(), anim.out);
    outUsable = true;
  }

  if (!combined) {
    if (outUsable && frames >= outSeqStart) {
      stage = rta::Stage::Out;
      useSeg = segOut.get();
      useSet = &anim.out;
      useMode = outMode;
      stageStart = outSeqStart;
      rawSlide = rawSlideOut;
      rawBlur = rawBlurOut;
    }
    if (stage != rta::Stage::Out) stage = anim.enableIn ? rta::Stage::In : rta::Stage::Settled;
  } else {
    stage = anim.enableIn ? rta::Stage::In : rta::Stage::Settled;
  }

  {
    // The timing mapping being right does not mean the animation is running:
    // a stale Start offset delays it just as effectively. Log what the animator
    // is actually being asked to do, so the two cannot be confused again.
    char buf[320];
    std::snprintf(buf, sizeof(buf),
                  "stage=%s in=%d out=%d start=%.1f dur=%.1f stagger=%.1f groups=%zu "
                  "stageStart=%.1f | clipFrame=%.1f",
                  stage == rta::Stage::Out ? "OUT" : (stage == rta::Stage::In ? "IN" : "SETTLED"),
                  anim.enableIn ? 1 : 0, anim.enableOut ? 1 : 0, anim.startTime, useSet->duration,
                  useSet->stagger, useSeg->groups.size(), stageStart, frames);
    rta::probeOnChange("anim", buf);
  }

  // Lengths are authored as a ratio so the motion looks identical at 1080p and
  // 4K. renderScale only tracks proxy scale, not timeline resolution, so it
  // cannot do this on its own: at a 4K timeline it is still 1.0 while the frame
  // and the text have both doubled.
  double lengthScale = args.renderScale.x;  // Pixels: authored at full res
  if (distanceUnits == 0) {
    lengthScale = srcView.height / 100.0;  // % of frame height
  } else if (distanceUnits == 1) {
    // % of text height: also independent of font size. Median group height is
    // a robust stand-in -- a mean would be dragged around by one tall word.
    std::vector<int> heights;
    heights.reserve(useSeg->groups.size());
    for (const auto& g : useSeg->groups) heights.push_back(g.bbox.height());
    if (!heights.empty()) {
      std::nth_element(heights.begin(), heights.begin() + heights.size() / 2, heights.end());
      lengthScale = std::max(1, heights[heights.size() / 2]) / 100.0;
    }
  }
  rta::StageSettings active = *useSet;
  active.slideDistance = rawSlide * lengthScale;
  active.startBlur = rawBlur * lengthScale;
  // tapCount looks at both stages, so keep them in step with the scaling.
  anim.in.startBlur = rawBlurIn * lengthScale;
  anim.out.startBlur = rawBlurOut * lengthScale;

  const std::vector<int> rank = rta::revealOrder(useSeg->groups, useSeg->lineCount, active);
  const int taps = rta::tapCount(anim);
  std::vector<rta::GroupTransform> transforms(useSeg->groups.size() * size_t(taps));

  if (combined && outUsable) {
    // Both stages share a grouping, so each group can be asked for both and take
    // whichever is further from settled.
    rta::StageSettings outActive = anim.out;
    outActive.slideDistance = rawSlideOut * lengthScale;
    outActive.startBlur = rawBlurOut * lengthScale;
    const std::vector<int> rankOut =
        rta::revealOrder(useSeg->groups, useSeg->lineCount, outActive);
    for (size_t i = 0; i < useSeg->groups.size(); ++i)
      rta::transformTapsCombined(rank[i], rankOut[i], frames, anim.startTime, outSeqStart, anim,
                                 active, outActive, &transforms[i * size_t(taps)]);
  } else {
    for (size_t i = 0; i < useSeg->groups.size(); ++i)
      rta::transformTaps(stage, rank[i], frames, stageStart, anim, active,
                         &transforms[i * size_t(taps)]);
  }

  // Room check for the indicator, using the group count we just measured.
  {
    const rta::FitReport fit = rta::checkFit(useSeg->groups.size(), anim, clipLength, haveLength);
    _fitOk.store(fit.ok(), std::memory_order_relaxed);

    // Publish for the overlay, which is where these are actually legible.
    double wt1 = 0.0, wt2 = 0.0;
    bool wok = true;
    try {
      timeLineGetBounds(wt1, wt2);
    } catch (...) {
      wok = false;
    }
    rta::WarningState w;
    w.sourceOffset = wok && wt1 < 0.0;
    w.clipTooShort = !fit.ok();
    rta::setWarningState(this, w);
  }

#if RTA_WITH_CUDA
  if (useCuda) {
    // Keep the device stencil in step with whichever segmentation this frame
    // uses; the two stages may group differently.
    std::lock_guard<std::mutex> lock(_cache.mutex);
    if (_cache.devSegMode != int(useMode)) {
      _cache.devSegValid = _cache.devSeg.upload(*useSeg);
      _cache.devSegMode = _cache.devSegValid ? int(useMode) : -1;
    }
  }
#endif

  {
    // What the animator actually produced for the first group. If the composite
    // draws nothing while the diagnostics overlay still appears, this is the
    // only place left that can explain it.
    const rta::GroupTransform& g0 = transforms[0];
    int visibleCount = 0;
    for (size_t i = 0; i < useSeg->groups.size(); ++i)
      if (transforms[i * size_t(taps)].visible) ++visibleCount;
    const double t0 = stageStart + double(rank.empty() ? 0 : rank[0]) * useSet->stagger;
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "clipFrame=%.1f g0: rank=%d t0=%.1f raw=%.3f vis=%d op=%.3f | visibleGroups=%d/%zu",
                  frames, rank.empty() ? -1 : rank[0], t0,
                  (frames - t0) / std::max(1e-6, useSet->duration), g0.visible ? 1 : 0, g0.opacity,
                  visibleCount, useSeg->groups.size());
    rta::probeOnChange("g0", buf);
  }

  const rta::RectI window = toLocal(args.renderWindow, common);
  const bool diagnostics = _showDiagnostics->getValueAtTime(args.time);

#if RTA_WITH_CUDA
  if (useCuda) {
    std::lock_guard<std::mutex> lock(_cache.mutex);
    if (!_cache.devSegValid) {
      trace.why = "cuda-devseg-invalid";
      return;
    }
    if (!rta::cudaCompositeGroups(dstView.data, dstView.rowStride, srcView.data, srcView.rowStride,
                                  dstView.width, dstView.height, _cache.devSeg, useSeg->groups,
                                  transforms, taps, window, _cache.scratch, args.pCudaStream)) {
      trace.why = "cuda-composite-failed";
      return;
    }
    if (diagnostics) {
      rta::cudaDrawDiagnostics(dstView.data, dstView.rowStride, srcView.data, srcView.rowStride,
                               dstView.width, dstView.height, useSeg->groups, window, 2,
                               args.pCudaStream);
    }
    trace.why = "ok-cuda";
    return;
  }
#endif

  rta::compositeGroups(dstView, srcView, *useSeg, transforms, taps, window);

  if (diagnostics) {
    // Draw the detected boxes over the *source* layout, not the animated one,
    // so the overlay answers "did detection work" independently of timing.
    for (int y = std::max(0, window.y1); y < std::min(dstView.height, window.y2); ++y) {
      const float* s = srcView.at(std::max(0, window.x1), y);
      float* d = dstView.at(std::max(0, window.x1), y);
      const int n = std::min(dstView.width, window.x2) - std::max(0, window.x1);
      for (int i = 0; i < n * 4; i += 4) {
        for (int c = 0; c < 4; ++c) d[i + c] = std::max(d[i + c], s[i + c] * 0.25f);
      }
    }
    rta::drawDiagnostics(dstView, *useSeg, 2);
  }
  trace.why = "ok-cpu";
}

// ---------------------------------------------------------------- factory

mDeclarePluginFactory(TextAnimatorFactory, {}, {});

void TextAnimatorFactory::describe(OFX::ImageEffectDescriptor& desc) {
  desc.setLabels(kPluginName, kPluginName, kPluginName);
  desc.setPluginGrouping(kPluginGrouping);
  desc.setPluginDescription(kPluginDescription);

  desc.addSupportedContext(OFX::eContextFilter);
  desc.addSupportedContext(OFX::eContextGeneral);
  desc.addSupportedBitDepth(OFX::eBitDepthFloat);

  desc.setSingleInstance(false);
  desc.setHostFrameThreading(false);
  desc.setSupportsMultiResolution(false);
  // Mandatory: word grouping is a whole-image measurement. Given a tile, the
  // detector would find different words in each tile.
  desc.setSupportsTiles(false);
  desc.setTemporalClipAccess(false);
  desc.setRenderTwiceAlways(false);
  desc.setSupportsMultipleClipDepths(false);
  desc.setRenderThreadSafety(OFX::eRenderInstanceSafe);

#if RTA_WITH_CUDA
  // Without this Resolve pulls every frame back to system memory to run us,
  // which is what stops a CPU plugin from playing back in realtime.
  desc.setSupportsCudaRender(true);
  desc.setSupportsCudaStream(true);
#endif

  // Plugin-level property, so it belongs in describe() rather than
  // describeInContext() -- the host caches the descriptor and would not
  // necessarily pick it up from the per-context pass.
  rta::registerOverlay(desc);
}

void TextAnimatorFactory::describeInContext(OFX::ImageEffectDescriptor& desc, OFX::ContextEnum) {
  OFX::ClipDescriptor* src = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
  src->addSupportedComponent(OFX::ePixelComponentRGBA);
  src->setTemporalClipAccess(false);
  src->setSupportsTiles(false);
  src->setIsMask(false);

  OFX::ClipDescriptor* dst = desc.defineClip(kOfxImageEffectOutputClipName);
  dst->addSupportedComponent(OFX::ePixelComponentRGBA);
  dst->setSupportsTiles(false);

  OFX::PageParamDescriptor* page = desc.definePageParam("Controls");

  {
    // Status line. Resolve draws a string parameter's label rather than its
    // value, so the text lives in the label and is rewritten at runtime.
    OFX::StringParamDescriptor* p = desc.defineStringParam("hostWarning");
    p->setStringType(OFX::eStringTypeLabel);
    p->setLabels("Timing OK", "Timing", "Timing");
    p->setHint(
        "(!) Source offset means this clip reaches back before its source start, "
        "so Resolve's render time follows the source rather than the trimmed "
        "clip and the reveal begins later than the clip's first frame.\n\n"
        "Fusion clips with a trimmed head do this; stills and PNGs do not.\n\n"
        "Fix it either by applying the effect inside the Fusion comp, or by "
        "lowering Start (frames) until the first word appears on the clip's "
        "first frame.");
    p->setDefault("");
    addParamUI(page, p);  // status only: never worth purging the cache for
  }
  {
    OFX::BooleanParamDescriptor* p = desc.defineBooleanParam("enableIn");
    p->setLabels("In Animation", "In", "Enable In Animation");
    p->setHint("Animate the text on. With both In and Out off the text simply sits there.");
    p->setDefault(true);
    addParam(page, p);
  }
  {
    OFX::BooleanParamDescriptor* p = desc.defineBooleanParam("enableOut");
    p->setLabels("Out Animation", "Out", "Enable Out Animation");
    p->setHint("Animate the text off, anchored to the end of the clip.");
    p->setDefault(false);
    addParam(page, p);
  }

  {
    OFX::ChoiceParamDescriptor* p = desc.defineChoiceParam("groupMode");
    p->setLabels("Animate By", "Animate By", "Animate By");
    p->setHint("What counts as one animated unit, detected from pixel gaps.");
    p->appendOption("Character");
    p->appendOption("Word");
    p->appendOption("Line");
    p->setDefault(1);
    addParam(page, p);
  }
  {
    OFX::ChoiceParamDescriptor* p = desc.defineChoiceParam("animation");
    p->setLabels("Animation", "Animation", "Animation");
    p->appendOption("Fade");
    p->appendOption("Slide + Fade");
    p->setDefault(1);
    addParam(page, p);
  }
  {
    OFX::ChoiceParamDescriptor* p = desc.defineChoiceParam("order");
    p->setLabels("Order", "Order", "Order");
    p->appendOption("Forward");
    p->appendOption("Reverse");
    p->appendOption("Center Out");
    p->appendOption("Random");
    p->setDefault(0);
    addParam(page, p);
  }
  {
    OFX::ChoiceParamDescriptor* p = desc.defineChoiceParam("lineOrder");
    p->setLabels("Line Order", "Line Order", "Line Order");
    p->setHint(
        "Which line reveals first. Words still read left to right within each "
        "line, unlike Order > Reverse which reverses those too.");
    p->appendOption("Top to Bottom");
    p->appendOption("Bottom to Top");
    p->setDefault(0);
    addParam(page, p);
  }
  {
    OFX::IntParamDescriptor* p = desc.defineIntParam("randomSeed");
    p->setLabels("Random Seed", "Seed", "Random Seed");
    p->setRange(0, 9999);
    p->setDisplayRange(0, 100);
    p->setDefault(0);
    addParam(page, p);
  }
  {
    OFX::ChoiceParamDescriptor* p = desc.defineChoiceParam("easing");
    p->setLabels("Easing", "Easing", "Easing");
    p->appendOption("Linear");
    p->appendOption("Smoothstep");
    p->appendOption("Cubic Out");
    p->appendOption("Back Out");
    p->appendOption("Custom (curve editor)");
    p->setDefault(2);
    addParam(page, p);
  }
  {
    OFX::BooleanParamDescriptor* p = desc.defineBooleanParam("showCurveEditor");
    p->setLabels("Show Curve Editor", "Curve Editor", "Show Curve Editor");
    p->setHint(
        "Draws the easing curve over the viewer. Set the Viewer's on-screen-control "
        "dropdown to \"Open FX Overlay\" to see it, then drag the two handles.");
    p->setDefault(false);
    addParamUI(page, p);
  }

  // The bezier control points. They are driven by dragging the on-screen
  // handles, but stay visible as numbers so a curve can be typed in or copied
  // between clips.
  {
    static const char* names[4] = {"easeX1", "easeY1", "easeX2", "easeY2"};
    static const char* labels[4] = {"Curve X1", "Curve Y1", "Curve X2", "Curve Y2"};
    static const double defs[4] = {0.25, 0.1, 0.25, 1.0};  // ~ CSS "ease"
    for (int i = 0; i < 4; ++i) {
      OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(names[i]);
      p->setLabels(labels[i], labels[i], labels[i]);
      // x is bounded to 0..1 or the curve stops being a function of time; y is
      // free so handles can be pulled past the rails for overshoot.
      if (i % 2 == 0)
        p->setRange(0.0, 1.0);
      else
        p->setRange(-10.0, 10.0);
      p->setDisplayRange(i % 2 == 0 ? 0.0 : -1.0, i % 2 == 0 ? 1.0 : 2.0);
      p->setDefault(defs[i]);
      addParam(page, p);
    }
  }
  {
    OFX::PushButtonParamDescriptor* p = desc.definePushButtonParam("setStartToPlayhead");
    p->setLabels("Set Start to Playhead", "Set Start", "Set Start to Playhead");
    p->setHint(
        "Park the playhead where the animation should begin -- normally the "
        "clip's first frame -- and click. Resolve does not report where a "
        "trimmed clip visually starts, so this captures the playhead in the "
        "same units the renderer uses; the two cancel out and the entrance "
        "begins exactly here however the clip is trimmed. Re-click after "
        "trimming the head.");
    // hidden: see originAt -- the boundary is already frame 0
  }
  {
    OFX::PushButtonParamDescriptor* p = desc.definePushButtonParam("setStartToClipStart");
    p->setLabels("Set Start to Clip Start", "Clip Start", "Set Start to Clip Start");
    p->setHint(
        "Alternative to the above that needs no playhead: scrub across the clip "
        "once, then click. Captures the earliest frame Resolve actually "
        "rendered, which is the clip's first frame.");
    // hidden: see originAt -- the boundary is already frame 0
  }
  {
    OFX::BooleanParamDescriptor* p = desc.defineBooleanParam("manualOrigin");
    p->setLabels("Use Manual Start", "Manual Start", "Use Manual Start Anchor");
    p->setHint(
        "Ticked automatically by the button above. Untick to fall back to the "
        "host's reported clip start, which is exact only on clips with no head "
        "handle.");
    p->setDefault(false);
    // hidden: see originAt -- the boundary is already frame 0
  }
  {
    OFX::DoubleParamDescriptor* p = desc.defineDoubleParam("originFrame");
    p->setLabels("Start Anchor (frame)", "Anchor", "Start Anchor Frame");
    p->setHint("The captured timeline frame that counts as frame 0. Editable and copyable.");
    p->setRange(-1000000.0, 1000000.0);
    p->setDisplayRange(0.0, 10000.0);
    p->setDefault(0.0);
    // hidden: see originAt -- the boundary is already frame 0
  }
  {
    OFX::DoubleParamDescriptor* p = desc.defineDoubleParam("startTime");
    p->setLabels("Start (frames)", "Start", "Start Time");
    p->setHint("Frames from the clip's first frame before the first unit appears.");
    p->setRange(-100000.0, 100000.0);
    p->setDisplayRange(0.0, 120.0);
    p->setDefault(0.0);
    addParam(page, p);
  }
  {
    OFX::DoubleParamDescriptor* p = desc.defineDoubleParam("duration");
    p->setLabels("Duration (frames)", "Duration", "Duration");
    p->setHint("How many frames a single unit takes to animate in.");
    p->setRange(0.01, 100000.0);
    p->setDisplayRange(1.0, 60.0);
    p->setDefault(12.0);
    addParam(page, p);
  }
  {
    OFX::DoubleParamDescriptor* p = desc.defineDoubleParam("stagger");
    p->setLabels("Stagger (frames)", "Stagger", "Stagger");
    p->setHint(
        "Frames between one unit starting and the next. Fractional values are "
        "allowed and are often what reads best.");
    p->setRange(0.0, 100000.0);
    p->setDisplayRange(0.0, 15.0);
    p->setDefault(2.0);
    addParam(page, p);
  }
  {
    OFX::ChoiceParamDescriptor* p = desc.defineChoiceParam("distanceUnits");
    p->setLabels("Distance Units", "Units", "Distance Units");
    p->setHint(
        "What Slide Distance and Start Blur are measured against. The two "
        "percentage modes keep the motion looking identical at 1080p and 4K; "
        "Pixels does not.");
    p->appendOption("% of Frame Height");
    p->appendOption("% of Text Height");
    p->appendOption("Pixels");
    p->setDefault(0);
    addParam(page, p);
  }
  {
    OFX::DoubleParamDescriptor* p = desc.defineDoubleParam("slideDistance");
    p->setLabels("Slide Distance", "Distance", "Slide Distance");
    p->setHint("How far a unit travels, in the chosen Distance Units.");
    p->setRange(-10000.0, 10000.0);
    p->setDisplayRange(0.0, 30.0);
    p->setDefault(4.0);  // ~40px at 1080p, the old pixel default
    addParam(page, p);
  }
  {
    OFX::DoubleParamDescriptor* p = desc.defineDoubleParam("slideAngle");
    p->setLabels("Slide Angle", "Angle", "Slide Angle");
    p->setHint("Direction the unit travels from. 90 rises from below.");
    p->setRange(-360.0, 360.0);
    p->setDisplayRange(0.0, 360.0);
    p->setDefault(90.0);
    addParam(page, p);
  }
  {
    OFX::DoubleParamDescriptor* p = desc.defineDoubleParam("startScale");
    p->setLabels("Start Scale", "Start Scale", "Start Scale");
    p->setRange(0.01, 10.0);
    p->setDisplayRange(0.2, 2.0);
    p->setDefault(1.0);
    addParam(page, p);
  }

  {
    OFX::DoubleParamDescriptor* p = desc.defineDoubleParam("startRotation");
    p->setLabels("Start Rotation", "Rotation", "Start Rotation");
    p->setHint("Degrees each unit is rotated by before it settles. Rotation is about its own centre.");
    p->setRange(-3600.0, 3600.0);
    p->setDisplayRange(-180.0, 180.0);
    p->setDefault(0.0);
    addParam(page, p);
  }
  {
    OFX::DoubleParamDescriptor* p = desc.defineDoubleParam("startBlur");
    p->setLabels("Start Blur", "Blur", "Start Blur");
    p->setHint(
        "Defocus radius at the start of a unit's animation, sharpening as it "
        "settles. Measured in the chosen Distance Units.");
    p->setRange(0.0, 500.0);
    p->setDisplayRange(0.0, 5.0);
    p->setDefault(0.0);
    addParam(page, p);
  }

  // ------------------------------------------------------------ out stage
  {
    OFX::GroupParamDescriptor* g = desc.defineGroupParam("outStage");
    g->setLabels("Out Animation", "Out Animation", "Out Animation");
    g->setOpen(false);

    {
      OFX::DoubleParamDescriptor* p = desc.defineDoubleParam("outOffset");
      p->setLabels("End Offset (frames)", "End Offset", "Out End Offset");
      p->setHint(
          "Frames before the END of the clip at which the out animation "
          "finishes. 30 means it is fully gone 30 frames before the clip ends. "
          "Anchored to the end, so it stays put when the clip is trimmed.");
      p->setRange(-100000.0, 100000.0);
      p->setDisplayRange(0.0, 120.0);
      p->setDefault(0.0);
      p->setParent(*g);
      addParam(page, p);
    }
    {
      OFX::PushButtonParamDescriptor* p = desc.definePushButtonParam("setOutEndToPlayhead");
      p->setLabels("Set End to Playhead", "Set End", "Set Out End to Playhead");
      p->setHint("Park the playhead where the exit should finish and click.");
      p->setParent(*g);
      // A push button holds no value, so it has no cache invalidation to set --
      // the parameter it writes carries that instead.
      addParamUI(page, p);
    }
    {
      OFX::BooleanParamDescriptor* p = desc.defineBooleanParam("linkOut");
      p->setLabels("Link to In", "Link", "Link Out to In");
      p->setHint("Reuse the In settings for the Out animation. Untick for an independent set.");
      p->setDefault(true);
      p->setParent(*g);
      addParam(page, p);
    }
    {
      OFX::BooleanParamDescriptor* p = desc.defineBooleanParam("mirrorOut");
      p->setLabels("Mirror", "Mirror", "Mirror Out");
      p->setHint(
          "Linked only. On, the text retreats the way it came: an entrance "
          "rising from below sinks back down. Off, it continues in the "
          "direction of travel and carries on upward.");
      p->setDefault(true);
      p->setParent(*g);
      addParam(page, p);
    }
    {
      OFX::DoubleParamDescriptor* p = desc.defineDoubleParam("clipLengthOverride");
      p->setLabels("Clip Length Override", "Clip Length", "Clip Length Override");
      p->setHint(
          "0 = ask the host. Set this only if the host does not report a clip "
          "length, in which case the Out animation has no end to anchor to and "
          "is skipped.");
      p->setRange(0.0, 1000000.0);
      p->setDisplayRange(0.0, 600.0);
      p->setDefault(0.0);
      p->setParent(*g);
      addParam(page, p);
    }

    // Used only when Link is off.
    {
      OFX::ChoiceParamDescriptor* p = desc.defineChoiceParam("outGroupMode");
      p->setLabels("Out Animate By", "Out By", "Out Animate By");
      p->setHint("Unlinked only. The Out stage may group differently from the In stage.");
      p->appendOption("Character");
      p->appendOption("Word");
      p->appendOption("Line");
      p->setDefault(1);
      p->setParent(*g);
      addParam(page, p);
    }
    {
      OFX::ChoiceParamDescriptor* p = desc.defineChoiceParam("outAnimation");
      p->setLabels("Out Animation Type", "Out Type", "Out Animation Type");
      p->appendOption("Fade");
      p->appendOption("Slide + Fade");
      p->setDefault(1);
      p->setParent(*g);
      addParam(page, p);
    }
    {
      OFX::ChoiceParamDescriptor* p = desc.defineChoiceParam("outEasing");
      p->setLabels("Out Easing", "Out Easing", "Out Easing");
      p->appendOption("Linear");
      p->appendOption("Smoothstep");
      p->appendOption("Cubic Out");
      p->appendOption("Back Out");
      p->appendOption("Custom (curve editor)");
      p->setDefault(2);
      p->setParent(*g);
      addParam(page, p);
    }
    {
      OFX::ChoiceParamDescriptor* p = desc.defineChoiceParam("outOrder");
      p->setLabels("Out Order", "Out Order", "Out Order");
      p->appendOption("Forward");
      p->appendOption("Reverse");
      p->appendOption("Center Out");
      p->appendOption("Random");
      p->setDefault(0);
      p->setParent(*g);
      addParam(page, p);
    }
    {
      OFX::ChoiceParamDescriptor* p = desc.defineChoiceParam("outLineOrder");
      p->setLabels("Out Line Order", "Out Lines", "Out Line Order");
      p->appendOption("Top to Bottom");
      p->appendOption("Bottom to Top");
      p->setDefault(0);
      p->setParent(*g);
      addParam(page, p);
    }
    {
      OFX::IntParamDescriptor* p = desc.defineIntParam("outRandomSeed");
      p->setLabels("Out Random Seed", "Out Seed", "Out Random Seed");
      p->setRange(0, 9999);
      p->setDisplayRange(0, 100);
      p->setDefault(0);
      p->setParent(*g);
      addParam(page, p);
    }
    {
      OFX::DoubleParamDescriptor* p = desc.defineDoubleParam("outDuration");
      p->setLabels("Out Duration (frames)", "Out Duration", "Out Duration");
      p->setRange(0.01, 100000.0);
      p->setDisplayRange(1.0, 60.0);
      p->setDefault(12.0);
      p->setParent(*g);
      addParam(page, p);
    }
    {
      OFX::DoubleParamDescriptor* p = desc.defineDoubleParam("outStagger");
      p->setLabels("Out Stagger (frames)", "Out Stagger", "Out Stagger");
      p->setRange(0.0, 100000.0);
      p->setDisplayRange(0.0, 15.0);
      p->setDefault(2.0);
      p->setParent(*g);
      addParam(page, p);
    }
    {
      OFX::DoubleParamDescriptor* p = desc.defineDoubleParam("outSlideDistance");
      p->setLabels("Out Slide Distance", "Out Distance", "Out Slide Distance");
      p->setHint("In the same Distance Units as the In stage.");
      p->setRange(-10000.0, 10000.0);
      p->setDisplayRange(0.0, 30.0);
      p->setDefault(4.0);
      p->setParent(*g);
      addParam(page, p);
    }
    {
      OFX::DoubleParamDescriptor* p = desc.defineDoubleParam("outSlideAngle");
      p->setLabels("Out Slide Angle", "Out Angle", "Out Slide Angle");
      p->setHint("Direction the unit departs towards. 90 sinks downward.");
      p->setRange(-360.0, 360.0);
      p->setDisplayRange(0.0, 360.0);
      p->setDefault(90.0);
      p->setParent(*g);
      addParam(page, p);
    }
    {
      OFX::DoubleParamDescriptor* p = desc.defineDoubleParam("outStartScale");
      p->setLabels("Out End Scale", "Out Scale", "Out End Scale");
      p->setRange(0.01, 10.0);
      p->setDisplayRange(0.2, 2.0);
      p->setDefault(1.0);
      p->setParent(*g);
      addParam(page, p);
    }
    {
      OFX::DoubleParamDescriptor* p = desc.defineDoubleParam("outStartRotation");
      p->setLabels("Out End Rotation", "Out Rotation", "Out End Rotation");
      p->setRange(-3600.0, 3600.0);
      p->setDisplayRange(-180.0, 180.0);
      p->setDefault(0.0);
      p->setParent(*g);
      addParam(page, p);
    }
    {
      OFX::DoubleParamDescriptor* p = desc.defineDoubleParam("outStartBlur");
      p->setLabels("Out End Blur", "Out Blur", "Out End Blur");
      p->setRange(0.0, 500.0);
      p->setDisplayRange(0.0, 5.0);
      p->setDefault(0.0);
      p->setParent(*g);
      addParam(page, p);
    }
  }

  {
    OFX::GroupParamDescriptor* g = desc.defineGroupParam("motion");
    g->setLabels("Motion Blur", "Motion Blur", "Motion Blur");
    g->setOpen(false);

    {
      OFX::BooleanParamDescriptor* p = desc.defineBooleanParam("motionBlur");
      p->setLabels("Motion Blur", "Motion Blur", "Motion Blur");
      p->setHint(
          "Supersamples each unit's transform across the open shutter, so "
          "movement, scaling and rotation all blur.");
      p->setDefault(false);
      p->setParent(*g);
      addParam(page, p);
    }
    {
      OFX::DoubleParamDescriptor* p = desc.defineDoubleParam("shutterAngle");
      p->setLabels("Shutter Angle", "Shutter", "Shutter Angle");
      p->setHint("Degrees of a frame the shutter is open. 180 matches a normal cine shutter.");
      p->setRange(0.0, 360.0);
      p->setDisplayRange(0.0, 360.0);
      p->setDefault(180.0);
      p->setParent(*g);
      addParam(page, p);
    }
    {
      OFX::IntParamDescriptor* p = desc.defineIntParam("blurSamples");
      p->setLabels("Samples", "Samples", "Blur Samples");
      p->setHint("Taps used for motion blur and start blur. Higher is smoother but slower.");
      p->setRange(2, 64);
      p->setDisplayRange(2, 32);
      p->setDefault(8);
      p->setParent(*g);
      addParam(page, p);
    }
  }

  {
    OFX::GroupParamDescriptor* g = desc.defineGroupParam("detection");
    g->setLabels("Detection", "Detection", "Detection");
    g->setOpen(false);

    {
      OFX::DoubleParamDescriptor* p = desc.defineDoubleParam("alphaThreshold");
      p->setLabels("Alpha Threshold", "Alpha", "Alpha Threshold");
      p->setHint("Alpha above this counts as text. Raise it if a soft shadow is bridging glyphs.");
      p->setRange(0.0, 1.0);
      p->setDisplayRange(0.0, 1.0);
      p->setDefault(0.15);
      p->setParent(*g);
      addParam(page, p);
    }
    {
      OFX::IntParamDescriptor* p = desc.defineIntParam("minBlobArea");
      p->setLabels("Min Blob Area", "Min Area", "Min Blob Area");
      p->setHint("Discards specks smaller than this many pixels.");
      p->setRange(1, 10000);
      p->setDisplayRange(1, 64);
      p->setDefault(4);
      p->setParent(*g);
      addParam(page, p);
    }
    {
      OFX::DoubleParamDescriptor* p = desc.defineDoubleParam("wordGapSensitivity");
      p->setLabels("Word Gap", "Word Gap", "Word Gap Sensitivity");
      p->setHint("Lower splits words more eagerly, higher merges them. Adjust if a font mis-splits.");
      p->setRange(0.05, 10.0);
      p->setDisplayRange(0.3, 3.0);
      p->setDefault(1.0);
      p->setParent(*g);
      addParam(page, p);
    }
    {
      OFX::IntParamDescriptor* p = desc.defineIntParam("bridgeRadius");
      p->setLabels("Bridge Radius", "Bridge", "Bridge Radius");
      p->setHint("Joins glyphs that nearly touch. Use for script fonts.");
      p->setRange(0, 32);
      p->setDisplayRange(0, 6);
      p->setDefault(0);
      p->setParent(*g);
      addParam(page, p);
    }
    {
      OFX::BooleanParamDescriptor* p = desc.defineBooleanParam("showDiagnostics");
      p->setLabels("Show Detection", "Show Detection", "Show Detection");
      p->setHint("Draws a coloured box around each detected unit over the static text.");
      p->setDefault(false);
      p->setParent(*g);
      addParam(page, p);
    }
  }
}

OFX::ImageEffect* TextAnimatorFactory::createInstance(OfxImageEffectHandle handle,
                                                      OFX::ContextEnum) {
  return new TextAnimatorPlugin(handle);
}

}  // namespace

void OFX::Plugin::getPluginIDs(OFX::PluginFactoryArray& ids) {
  static TextAnimatorFactory factory(kPluginIdentifier, 1, 0);
  ids.push_back(&factory);
}
