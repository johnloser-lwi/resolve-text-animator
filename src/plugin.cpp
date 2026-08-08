// OFX host glue. Everything interesting lives in segmentation/animator/
// compositor; this file only fetches images, reads parameters, and maps OFX
// coordinates into the flat local space the core code works in.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <vector>

#include "ofxsImageEffect.h"
#include "ofxsMultiThread.h"

#include "animator.h"
#include "clip_time.h"
#include "compositor.h"
#include "cuda_compositor.h"
#include "diagnostics.h"
#include "interact/overlay.h"
#include "segmentation.h"

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
  rta::DetectParams params;
  std::shared_ptr<const rta::Segmentation> seg;
#if RTA_WITH_CUDA
  // Device copy of the same segmentation, uploaded whenever `seg` is replaced.
  rta::CudaSegmentation devSeg;
  rta::CudaScratch scratch;
  bool devSegValid = false;
#endif
};

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
    _easeX1 = fetchDoubleParam("easeX1");
    _easeY1 = fetchDoubleParam("easeY1");
    _easeX2 = fetchDoubleParam("easeX2");
    _easeY2 = fetchDoubleParam("easeY2");

    _alphaThreshold = fetchDoubleParam("alphaThreshold");
    _minBlobArea = fetchIntParam("minBlobArea");
    _wordGapSensitivity = fetchDoubleParam("wordGapSensitivity");
    _bridgeRadius = fetchIntParam("bridgeRadius");
    _showDiagnostics = fetchBooleanParam("showDiagnostics");
  }

  void render(const OFX::RenderArguments& args) override;

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

  AnalysisCache _cache;
};

void TextAnimatorPlugin::render(const OFX::RenderArguments& args) {
  std::unique_ptr<OFX::Image> dst(_dstClip->fetchImage(args.time));
  if (!dst.get()) return;

  if (dst->getPixelDepth() != OFX::eBitDepthFloat ||
      dst->getPixelComponents() != OFX::ePixelComponentRGBA) {
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
    return;
  }

  // Work in the region both images actually cover.
  const OfxRectI srcBounds = src->getBounds();
  OfxRectI common{std::max(srcBounds.x1, dstBounds.x1), std::max(srcBounds.y1, dstBounds.y1),
                  std::min(srcBounds.x2, dstBounds.x2), std::min(srcBounds.y2, dstBounds.y2)};
  if (common.x2 <= common.x1 || common.y2 <= common.y1) return;

  const rta::ImageView srcView = makeView(src.get(), common);
  const rta::ImageView dstView = makeView(dst.get(), common);
  if (!srcView.valid() || !dstView.valid()) return;

  // ------------------------------------------------------------ parameters
  rta::DetectParams det;
  det.alphaThreshold = float(_alphaThreshold->getValueAtTime(args.time));
  det.minBlobArea = std::max(1, _minBlobArea->getValueAtTime(args.time));
  det.wordGapSensitivity = float(_wordGapSensitivity->getValueAtTime(args.time));
  det.bridgeRadius = std::max(0, _bridgeRadius->getValueAtTime(args.time));
  det.mode = rta::GroupMode(choiceAt(_groupMode, args.time, 0, 2));

  rta::AnimParams anim;
  anim.animation = rta::Animation(choiceAt(_animation, args.time, 0, 1));
  anim.easing = rta::Easing(choiceAt(_easing, args.time, 0, 4));
  anim.bezier.x1 = float(_easeX1->getValueAtTime(args.time));
  anim.bezier.y1 = float(_easeY1->getValueAtTime(args.time));
  anim.bezier.x2 = float(_easeX2->getValueAtTime(args.time));
  anim.bezier.y2 = float(_easeY2->getValueAtTime(args.time));
  anim.order = rta::Order(choiceAt(_order, args.time, 0, 3));
  anim.lineOrder = rta::LineOrder(choiceAt(_lineOrder, args.time, 0, 1));
  anim.randomSeed = _randomSeed->getValueAtTime(args.time);
  anim.startTime = _startTime->getValueAtTime(args.time);
  anim.duration = _duration->getValueAtTime(args.time);
  anim.stagger = _stagger->getValueAtTime(args.time);
  anim.slideAngle = _slideAngle->getValueAtTime(args.time);
  anim.startScale = _startScale->getValueAtTime(args.time);
  // Slide distance is authored in full-resolution pixels, so scale it to keep
  // the motion identical at proxy, 1080p and 4K.
  anim.startRotation = _startRotation->getValueAtTime(args.time);
  // Slide distance and blur are lengths, so they are converted below once the
  // segmentation is known -- "% of text height" needs the measured glyphs.
  const double rawSlide = _slideDistance->getValueAtTime(args.time);
  const double rawBlur = _startBlur->getValueAtTime(args.time);
  const int distanceUnits = choiceAt(_distanceUnits, args.time, 0, 2);
  anim.motionBlur = _motionBlur->getValueAtTime(args.time);
  anim.shutterAngle = _shutterAngle->getValueAtTime(args.time);
  anim.blurSamples = _blurSamples->getValueAtTime(args.time);

  // --------------------------------------------------------------- analysis
  uint64_t hash = 0;
#if RTA_WITH_CUDA
  if (useCuda) {
    bool ok = false;
    std::lock_guard<std::mutex> lock(_cache.mutex);
    hash = rta::cudaHashAlpha(srcView.data, srcView.rowStride, srcView.width, srcView.height,
                              det.alphaThreshold, _cache.scratch, args.pCudaStream, &ok);
    if (!ok) return;
  } else
#endif
  {
    hash = rta::hashAlpha(srcView, det.alphaThreshold);
  }

  std::shared_ptr<const rta::Segmentation> seg;
  {
    std::lock_guard<std::mutex> lock(_cache.mutex);
    const bool hit = _cache.seg && _cache.hash == hash && _cache.width == srcView.width &&
                     _cache.height == srcView.height && _cache.params == det;
    if (!hit) {
#if RTA_WITH_CUDA
      if (useCuda) {
        // The only full-frame readback in the whole plugin, and it happens once
        // per title change rather than once per frame.
        std::vector<float> host;
        if (!rta::cudaDownload(host, srcView.data, srcView.rowStride, srcView.width,
                               srcView.height))
          return;
        const std::ptrdiff_t absStride =
            srcView.rowStride < 0 ? -srcView.rowStride : srcView.rowStride;
        rta::ImageView hostView{host.data() + absStride * (srcView.height - 1), srcView.width,
                                srcView.height, -absStride};
        _cache.seg = std::make_shared<const rta::Segmentation>(rta::segment(hostView, det));
        _cache.devSegValid = _cache.devSeg.upload(*_cache.seg);
      } else
#endif
      {
        _cache.seg = std::make_shared<const rta::Segmentation>(rta::segment(srcView, det));
      }
      _cache.hash = hash;
      _cache.width = srcView.width;
      _cache.height = srcView.height;
      _cache.params = det;
    }
    seg = _cache.seg;
  }
  if (seg->empty()) return;

  // ---------------------------------------------------------------- animate
  double fps = _srcClip->getFrameRate();
  if (!(fps > 0.0)) fps = 25.0;
  // Clip-relative, so trimming the clip's head re-anchors the animation to the
  // new first frame instead of stranding it at a fixed timeline position.
  // getFrameRange() is deliberately not used here: Resolve returns a 1000-minute
  // sentinel from it rather than the clip's extent. See clip_time.h.
  const double seconds = rta::toClipTime(this, args.time) / fps;
  anim.frameDuration = 1.0 / fps;  // shutter interval is a fraction of a frame

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
    heights.reserve(seg->groups.size());
    for (const auto& g : seg->groups) heights.push_back(g.bbox.height());
    if (!heights.empty()) {
      std::nth_element(heights.begin(), heights.begin() + heights.size() / 2, heights.end());
      lengthScale = std::max(1, heights[heights.size() / 2]) / 100.0;
    }
  }
  anim.slideDistance = rawSlide * lengthScale;
  anim.startBlur = rawBlur * lengthScale;

  const std::vector<int> rank = rta::revealOrder(seg->groups, seg->lineCount, anim);
  const int taps = rta::tapCount(anim);
  std::vector<rta::GroupTransform> transforms(seg->groups.size() * size_t(taps));
  for (size_t i = 0; i < seg->groups.size(); ++i)
    rta::transformTaps(rank[i], seconds, anim, &transforms[i * size_t(taps)]);

  const rta::RectI window = toLocal(args.renderWindow, common);
  const bool diagnostics = _showDiagnostics->getValueAtTime(args.time);

#if RTA_WITH_CUDA
  if (useCuda) {
    std::lock_guard<std::mutex> lock(_cache.mutex);
    if (!_cache.devSegValid) return;
    if (!rta::cudaCompositeGroups(dstView.data, dstView.rowStride, srcView.data, srcView.rowStride,
                                  dstView.width, dstView.height, _cache.devSeg, seg->groups,
                                  transforms, taps, window, _cache.scratch, args.pCudaStream))
      return;
    if (diagnostics) {
      rta::cudaDrawDiagnostics(dstView.data, dstView.rowStride, srcView.data, srcView.rowStride,
                               dstView.width, dstView.height, seg->groups, window, 2,
                               args.pCudaStream);
    }
    return;
  }
#endif

  rta::compositeGroups(dstView, srcView, *seg, transforms, taps, window);

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
    rta::drawDiagnostics(dstView, *seg, 2);
  }
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
    OFX::ChoiceParamDescriptor* p = desc.defineChoiceParam("groupMode");
    p->setLabels("Animate By", "Animate By", "Animate By");
    p->setHint("What counts as one animated unit, detected from pixel gaps.");
    p->appendOption("Character");
    p->appendOption("Word");
    p->appendOption("Line");
    p->setDefault(1);
    page->addChild(*p);
  }
  {
    OFX::ChoiceParamDescriptor* p = desc.defineChoiceParam("animation");
    p->setLabels("Animation", "Animation", "Animation");
    p->appendOption("Fade");
    p->appendOption("Slide + Fade");
    p->setDefault(1);
    page->addChild(*p);
  }
  {
    OFX::ChoiceParamDescriptor* p = desc.defineChoiceParam("order");
    p->setLabels("Order", "Order", "Order");
    p->appendOption("Forward");
    p->appendOption("Reverse");
    p->appendOption("Center Out");
    p->appendOption("Random");
    p->setDefault(0);
    page->addChild(*p);
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
    page->addChild(*p);
  }
  {
    OFX::IntParamDescriptor* p = desc.defineIntParam("randomSeed");
    p->setLabels("Random Seed", "Seed", "Random Seed");
    p->setRange(0, 9999);
    p->setDisplayRange(0, 100);
    p->setDefault(0);
    page->addChild(*p);
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
    page->addChild(*p);
  }
  {
    OFX::BooleanParamDescriptor* p = desc.defineBooleanParam("showCurveEditor");
    p->setLabels("Show Curve Editor", "Curve Editor", "Show Curve Editor");
    p->setHint(
        "Draws the easing curve over the viewer. Set the Viewer's on-screen-control "
        "dropdown to \"Open FX Overlay\" to see it, then drag the two handles.");
    p->setDefault(false);
    page->addChild(*p);
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
      page->addChild(*p);
    }
  }
  {
    OFX::DoubleParamDescriptor* p = desc.defineDoubleParam("startTime");
    p->setLabels("Start (s)", "Start", "Start Time");
    p->setHint("Seconds from the start of the clip before the first unit appears.");
    p->setRange(0.0, 1000.0);
    p->setDisplayRange(0.0, 5.0);
    p->setDefault(0.0);
    page->addChild(*p);
  }
  {
    OFX::DoubleParamDescriptor* p = desc.defineDoubleParam("duration");
    p->setLabels("Duration (s)", "Duration", "Duration");
    p->setHint("How long a single unit takes to animate in.");
    p->setRange(0.001, 100.0);
    p->setDisplayRange(0.05, 2.0);
    p->setDefault(0.5);
    page->addChild(*p);
  }
  {
    OFX::DoubleParamDescriptor* p = desc.defineDoubleParam("stagger");
    p->setLabels("Stagger (s)", "Stagger", "Stagger");
    p->setHint("Delay between one unit starting and the next.");
    p->setRange(0.0, 100.0);
    p->setDisplayRange(0.0, 0.5);
    p->setDefault(0.06);
    page->addChild(*p);
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
    page->addChild(*p);
  }
  {
    OFX::DoubleParamDescriptor* p = desc.defineDoubleParam("slideDistance");
    p->setLabels("Slide Distance", "Distance", "Slide Distance");
    p->setHint("How far a unit travels, in the chosen Distance Units.");
    p->setRange(-10000.0, 10000.0);
    p->setDisplayRange(0.0, 30.0);
    p->setDefault(4.0);  // ~40px at 1080p, the old pixel default
    page->addChild(*p);
  }
  {
    OFX::DoubleParamDescriptor* p = desc.defineDoubleParam("slideAngle");
    p->setLabels("Slide Angle", "Angle", "Slide Angle");
    p->setHint("Direction the unit travels from. 90 rises from below.");
    p->setRange(-360.0, 360.0);
    p->setDisplayRange(0.0, 360.0);
    p->setDefault(90.0);
    page->addChild(*p);
  }
  {
    OFX::DoubleParamDescriptor* p = desc.defineDoubleParam("startScale");
    p->setLabels("Start Scale", "Start Scale", "Start Scale");
    p->setRange(0.01, 10.0);
    p->setDisplayRange(0.2, 2.0);
    p->setDefault(1.0);
    page->addChild(*p);
  }

  {
    OFX::DoubleParamDescriptor* p = desc.defineDoubleParam("startRotation");
    p->setLabels("Start Rotation", "Rotation", "Start Rotation");
    p->setHint("Degrees each unit is rotated by before it settles. Rotation is about its own centre.");
    p->setRange(-3600.0, 3600.0);
    p->setDisplayRange(-180.0, 180.0);
    p->setDefault(0.0);
    page->addChild(*p);
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
    page->addChild(*p);
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
      page->addChild(*p);
    }
    {
      OFX::DoubleParamDescriptor* p = desc.defineDoubleParam("shutterAngle");
      p->setLabels("Shutter Angle", "Shutter", "Shutter Angle");
      p->setHint("Degrees of a frame the shutter is open. 180 matches a normal cine shutter.");
      p->setRange(0.0, 360.0);
      p->setDisplayRange(0.0, 360.0);
      p->setDefault(180.0);
      p->setParent(*g);
      page->addChild(*p);
    }
    {
      OFX::IntParamDescriptor* p = desc.defineIntParam("blurSamples");
      p->setLabels("Samples", "Samples", "Blur Samples");
      p->setHint("Taps used for motion blur and start blur. Higher is smoother but slower.");
      p->setRange(2, 64);
      p->setDisplayRange(2, 32);
      p->setDefault(8);
      p->setParent(*g);
      page->addChild(*p);
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
      page->addChild(*p);
    }
    {
      OFX::IntParamDescriptor* p = desc.defineIntParam("minBlobArea");
      p->setLabels("Min Blob Area", "Min Area", "Min Blob Area");
      p->setHint("Discards specks smaller than this many pixels.");
      p->setRange(1, 10000);
      p->setDisplayRange(1, 64);
      p->setDefault(4);
      p->setParent(*g);
      page->addChild(*p);
    }
    {
      OFX::DoubleParamDescriptor* p = desc.defineDoubleParam("wordGapSensitivity");
      p->setLabels("Word Gap", "Word Gap", "Word Gap Sensitivity");
      p->setHint("Lower splits words more eagerly, higher merges them. Adjust if a font mis-splits.");
      p->setRange(0.05, 10.0);
      p->setDisplayRange(0.3, 3.0);
      p->setDefault(1.0);
      p->setParent(*g);
      page->addChild(*p);
    }
    {
      OFX::IntParamDescriptor* p = desc.defineIntParam("bridgeRadius");
      p->setLabels("Bridge Radius", "Bridge", "Bridge Radius");
      p->setHint("Joins glyphs that nearly touch. Use for script fonts.");
      p->setRange(0, 32);
      p->setDisplayRange(0, 6);
      p->setDefault(0);
      p->setParent(*g);
      page->addChild(*p);
    }
    {
      OFX::BooleanParamDescriptor* p = desc.defineBooleanParam("showDiagnostics");
      p->setLabels("Show Detection", "Show Detection", "Show Detection");
      p->setHint("Draws a coloured box around each detected unit over the static text.");
      p->setDefault(false);
      p->setParent(*g);
      page->addChild(*p);
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
