// Standalone segmentation harness.
//
//   segtest <in.png> <out.png> [--mode char|word|line] [--gap N] [--alpha N]
//           [--bridge N] [--minarea N]
//
// Runs the exact segmentation code the plugin uses and writes a diagnostic PNG.
// Tuning here takes seconds; tuning inside Resolve takes a relaunch each time.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "animator.h"
#include "compositor.h"
#include "diagnostics.h"
#include "segmentation.h"

namespace {

// Renders `count` evenly-spaced frames of the reveal into one contact sheet,
// cropped to the text plus its travel distance. Verifies the animator and
// compositor without a host in the loop.
std::vector<float> renderStrip(const rta::ImageView& src, const rta::Segmentation& seg,
                               const rta::AnimParams& anim, int count, double atFraction,
                               int& outW, int& outH) {
  rta::RectI area;
  for (const auto& g : seg.groups) area.unionWith(g.bbox);
  area.grow(int(anim.slideDistance) + 16);
  area.x1 = std::max(0, area.x1);
  area.y1 = std::max(0, area.y1);
  area.x2 = std::min(src.width, area.x2);
  area.y2 = std::min(src.height, area.y2);

  const int tw = area.width(), th = area.height();
  outW = tw;
  outH = th * count;
  std::vector<float> sheet(size_t(outW) * outH * 4, 0.0f);

  std::vector<float> frame(size_t(src.width) * src.height * 4, 0.0f);
  rta::ImageView fv{frame.data(), src.width, src.height, std::ptrdiff_t(src.width) * 4};

  const std::vector<int> rank = rta::revealOrder(seg.groups, seg.lineCount, anim);
  const double total = rta::totalDuration(seg.groups.size(), anim);

  const int taps = rta::tapCount(anim);
  for (int i = 0; i < count; ++i) {
    const double t = atFraction >= 0.0
                         ? total * atFraction
                         : total * double(i) / double(std::max(1, count - 1));
    std::vector<rta::GroupTransform> xf(seg.groups.size() * size_t(taps));
    for (size_t g = 0; g < seg.groups.size(); ++g)
      rta::transformTaps(rank[g], t, anim, &xf[g * size_t(taps)]);
    rta::compositeGroups(fv, src, seg, xf, taps, rta::RectI{0, 0, src.width, src.height});

    for (int y = 0; y < th; ++y) {
      const float* s = fv.at(area.x1, area.y1 + y);
      float* d = &sheet[(size_t(i) * th + y) * size_t(outW) * 4];
      std::copy(s, s + size_t(tw) * 4, d);
    }
  }
  return sheet;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: segtest <in.png> <out.png> [--mode char|word|line]\n"
                 "  detection: --gap N --alpha N --bridge N --minarea N\n"
                 "  animation: --strip N --at FRAC --stagger F --dur F --slide PX\n"
                 "             --angle DEG --easing 0-4 --order 0-3 --lineorder 0|1\n"
                 "             --rotate DEG --blur PX --mblur 0|1 --shutter DEG --samples N\n"
                 "             --bez x1,y1,x2,y2\n"
                 "  (--stagger and --dur are in FRAMES, matching the plugin)\n");
    return 2;
  }

  rta::DetectParams p;
  rta::AnimParams anim;
  int strip = 0;
  double atFraction = -1.0;
  for (int i = 3; i < argc; ++i) {
    const char* a = argv[i];
    auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : "0"; };
    if (!std::strcmp(a, "--mode")) {
      const char* m = next();
      p.mode = !std::strcmp(m, "char")   ? rta::GroupMode::Character
               : !std::strcmp(m, "line") ? rta::GroupMode::Line
                                         : rta::GroupMode::Word;
    } else if (!std::strcmp(a, "--gap")) {
      p.wordGapSensitivity = float(std::atof(next()));
    } else if (!std::strcmp(a, "--alpha")) {
      p.alphaThreshold = float(std::atof(next()));
    } else if (!std::strcmp(a, "--bridge")) {
      p.bridgeRadius = std::atoi(next());
    } else if (!std::strcmp(a, "--minarea")) {
      p.minBlobArea = std::atoi(next());
    } else if (!std::strcmp(a, "--strip")) {
      strip = std::atoi(next());
    } else if (!std::strcmp(a, "--stagger")) {
      anim.stagger = std::atof(next());
    } else if (!std::strcmp(a, "--dur")) {
      anim.duration = std::atof(next());
    } else if (!std::strcmp(a, "--slide")) {
      anim.slideDistance = std::atof(next());
    } else if (!std::strcmp(a, "--angle")) {
      anim.slideAngle = std::atof(next());
    } else if (!std::strcmp(a, "--easing")) {
      anim.easing = rta::Easing(std::atoi(next()));
    } else if (!std::strcmp(a, "--order")) {
      anim.order = rta::Order(std::atoi(next()));
    } else if (!std::strcmp(a, "--lineorder")) {
      anim.lineOrder = rta::LineOrder(std::atoi(next()));
    } else if (!std::strcmp(a, "--rotate")) {
      anim.startRotation = std::atof(next());
    } else if (!std::strcmp(a, "--blur")) {
      anim.startBlur = std::atof(next());
    } else if (!std::strcmp(a, "--mblur")) {
      anim.motionBlur = std::atoi(next()) != 0;
    } else if (!std::strcmp(a, "--shutter")) {
      anim.shutterAngle = std::atof(next());
    } else if (!std::strcmp(a, "--samples")) {
      anim.blurSamples = std::atoi(next());
    } else if (!std::strcmp(a, "--bez")) {
      // x1,y1,x2,y2 -- same control points the on-screen curve editor writes.
      std::sscanf(next(), "%f,%f,%f,%f", &anim.bezier.x1, &anim.bezier.y1, &anim.bezier.x2,
                  &anim.bezier.y2);
      anim.easing = rta::Easing::Custom;
    } else if (!std::strcmp(a, "--at")) {
      // Render a single frame at this fraction of the total duration, instead
      // of a strip -- for landing exactly on an overshoot.
      atFraction = std::atof(next());
    } else {
      std::fprintf(stderr, "unknown option: %s\n", a);
      return 2;
    }
  }

  int w = 0, h = 0, n = 0;
  unsigned char* png = stbi_load(argv[1], &w, &h, &n, 4);
  if (!png) {
    std::fprintf(stderr, "failed to read %s: %s\n", argv[1], stbi_failure_reason());
    return 1;
  }

  // stb gives straight (unassociated) 8-bit RGBA; the pipeline wants
  // premultiplied float, matching what Resolve hands the plugin.
  std::vector<float> buf(size_t(w) * h * 4);
  for (size_t i = 0; i < size_t(w) * h; ++i) {
    const float a = png[i * 4 + 3] / 255.0f;
    buf[i * 4 + 0] = (png[i * 4 + 0] / 255.0f) * a;
    buf[i * 4 + 1] = (png[i * 4 + 1] / 255.0f) * a;
    buf[i * 4 + 2] = (png[i * 4 + 2] / 255.0f) * a;
    buf[i * 4 + 3] = a;
  }
  stbi_image_free(png);

  rta::ImageView view{buf.data(), w, h, std::ptrdiff_t(w) * 4};
  const rta::Segmentation seg = rta::segment(view, p);

  std::printf("%dx%d  lines=%d  groups=%zu\n", w, h, seg.lineCount, seg.groups.size());
  for (size_t i = 0; i < seg.groups.size(); ++i) {
    const rta::Group& g = seg.groups[i];
    std::printf("  [%2zu] line=%d idx=%d  x=%d..%d y=%d..%d  (%dx%d) labels=%zu\n", i, g.line,
                g.indexInLine, g.bbox.x1, g.bbox.x2, g.bbox.y1, g.bbox.y2, g.bbox.width(),
                g.bbox.height(), g.labels.size());
  }

  int ow = w, oh = h;
  std::vector<float> strippedBuf;
  const float* result = buf.data();

  if (strip > 1) {
    strippedBuf = renderStrip(view, seg, anim, strip, atFraction, ow, oh);
    result = strippedBuf.data();
    std::printf("strip: %d samples over %.1f frames\n", strip,
                rta::totalDuration(seg.groups.size(), anim));
  } else {
    rta::drawDiagnostics(view, seg, 2);
  }

  // Back to straight 8-bit over a dark grey so boxes are visible in any viewer.
  std::vector<unsigned char> out(size_t(ow) * oh * 4);
  for (size_t i = 0; i < size_t(ow) * oh; ++i) {
    const float a = result[i * 4 + 3];
    for (int c = 0; c < 3; ++c) {
      const float v = result[i * 4 + c] + 0.15f * (1.0f - a);  // over dark grey
      out[i * 4 + c] = (unsigned char)(v < 0 ? 0 : v > 1 ? 255 : v * 255.0f + 0.5f);
    }
    out[i * 4 + 3] = 255;
  }
  if (!stbi_write_png(argv[2], ow, oh, 4, out.data(), ow * 4)) {
    std::fprintf(stderr, "failed to write %s\n", argv[2]);
    return 1;
  }
  std::printf("wrote %s\n", argv[2]);
  return 0;
}
