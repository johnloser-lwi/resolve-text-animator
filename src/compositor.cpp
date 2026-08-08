// All coordinates here are local to the image buffers: (0,0) is the first
// pixel of both src and dst, which the plugin guarantees by disabling tiling
// and multi-resolution so the two images always share bounds.
#include "compositor.h"

#include <algorithm>
#include <cmath>

#include "transform_geom.h"

namespace rta {
namespace {

RectI intersect(const RectI& a, const RectI& b) {
  RectI r{std::max(a.x1, b.x1), std::max(a.y1, b.y1), std::min(a.x2, b.x2), std::min(a.y2, b.y2)};
  if (r.empty()) return RectI{0, 0, 0, 0};
  return r;
}

// Bilinear tap that reads only pixels belonging to `group`. Masking before
// filtering is what keeps antialiased glyph edges intact and stops a moving
// word from smearing in a fragment of its neighbour.
inline void sampleMasked(const ImageView& src, const Segmentation& seg, int group, float fx,
                         float fy, float out[4]) {
  out[0] = out[1] = out[2] = out[3] = 0.0f;

  const int x0 = int(std::floor(fx));
  const int y0 = int(std::floor(fy));
  const float tx = fx - float(x0);
  const float ty = fy - float(y0);

  const float wx[2] = {1.0f - tx, tx};
  const float wy[2] = {1.0f - ty, ty};

  for (int j = 0; j < 2; ++j) {
    const int sy = y0 + j;
    if (sy < 0 || sy >= src.height) continue;
    for (int i = 0; i < 2; ++i) {
      const int sx = x0 + i;
      if (sx < 0 || sx >= src.width) continue;

      const int32_t lab = seg.labelImage[size_t(sy) * seg.width + sx];
      if (lab == 0 || seg.labelToGroup[lab] != group) continue;

      const float wgt = wx[i] * wy[j];
      if (wgt <= 0.0f) continue;
      const float* p = src.at(sx, sy);
      out[0] += p[0] * wgt;
      out[1] += p[1] * wgt;
      out[2] += p[2] * wgt;
      out[3] += p[3] * wgt;
    }
  }
}

}  // namespace

void compositeGroups(const ImageView& dst, const ImageView& src, const Segmentation& seg,
                     const std::vector<GroupTransform>& transforms, int taps,
                     const RectI& renderWindow) {
  if (!dst.valid()) return;

  const RectI dstRect{0, 0, dst.width, dst.height};
  const RectI window = intersect(renderWindow, dstRect);
  if (window.empty()) return;

  // Start from transparent: the source title is deliberately not drawn.
  for (int y = window.y1; y < window.y2; ++y) {
    float* row = dst.at(window.x1, y);
    std::fill(row, row + size_t(window.width()) * 4, 0.0f);
  }

  taps = std::max(1, taps);
  if (!src.valid() || seg.empty() || transforms.size() != seg.groups.size() * size_t(taps)) return;

  std::vector<TapGeom> geom(taps);

  for (size_t gi = 0; gi < seg.groups.size(); ++gi) {
    const GroupTransform* tg = &transforms[gi * size_t(taps)];

    const RectI& b = seg.groups[gi].bbox;
    const float cx = 0.5f * float(b.x1 + b.x2);
    const float cy = 0.5f * float(b.y1 + b.y2);

    // Union of every tap's footprint, so a fast-moving or spinning group is
    // never clipped to where it happens to be at the frame's midpoint.
    RectI dest{};
    int live = 0;
    for (int k = 0; k < taps; ++k) {
      geom[k] = TapGeom{tg[k], cx, cy};
      if (!tg[k].visible || tg[k].opacity <= 0.0f) continue;
      ++live;
      dest.unionWith(tapBounds(tg[k], b, cx, cy));
    }
    if (live == 0) continue;

    dest.grow(2);  // room for the bilinear footprint
    dest = intersect(dest, window);
    if (dest.empty()) continue;

    const float invTaps = 1.0f / float(taps);

    for (int y = dest.y1; y < dest.y2; ++y) {
      float* out = dst.at(dest.x1, y);
      for (int x = dest.x1; x < dest.x2; ++x, out += 4) {
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int k = 0; k < taps; ++k) {
          const GroupTransform& t = geom[k].t;
          if (!t.visible || t.opacity <= 0.0f) continue;

          // Defocus is a jitter of the destination sample position, which is
          // equivalent to blurring the composited result.
          float jx = 0.0f, jy = 0.0f;
          if (t.blur > 0.0f) {
            discOffset(k, taps, &jx, &jy);
            jx *= t.blur;
            jy *= t.blur;
          }

          float fx, fy;
          geom[k].inverse(float(x) + jx, float(y) + jy, &fx, &fy);

          float smp[4];
          sampleMasked(src, seg, int(gi), fx, fy, smp);
          if (smp[3] <= 0.0f) continue;

          acc[0] += smp[0] * t.opacity;
          acc[1] += smp[1] * t.opacity;
          acc[2] += smp[2] * t.opacity;
          acc[3] += smp[3] * t.opacity;
        }

        if (acc[3] <= 0.0f) continue;

        // Premultiplied, so fading is a plain scale of all four channels.
        const float a = acc[3] * invTaps;
        const float ia = 1.0f - a;
        out[0] = acc[0] * invTaps + out[0] * ia;
        out[1] = acc[1] * invTaps + out[1] * ia;
        out[2] = acc[2] * invTaps + out[2] * ia;
        out[3] = a + out[3] * ia;
      }
    }
  }
}

}  // namespace rta
