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

// Separable box blur, run three times, which is the usual stand-in for a
// Gaussian and is indistinguishable from one at these radii.
//
// Cost does not depend on the radius: each output pixel is one subtraction of
// two prefix sums. That is the whole reason for the change -- the defocus used
// to be N jittered copies of the sprite borrowed from the motion-blur taps, so
// it cost a full resample per tap AND showed every one of them as a distinct
// ghost until the sample count was pushed high enough to hide them.
//
// Runs on PREMULTIPLIED pixels, which is what makes a plain average correct:
// colour is already weighted by coverage, so a transparent neighbour
// contributes nothing rather than dragging black in.
void boxPass(const float* in, float* out, int w, int h, int r, bool horizontal) {
  const int n = horizontal ? w : h;
  const int lines = horizontal ? h : w;
  const std::ptrdiff_t step = horizontal ? 4 : std::ptrdiff_t(w) * 4;
  const std::ptrdiff_t lineStep = horizontal ? std::ptrdiff_t(w) * 4 : 4;
  const float inv = 1.0f / float(2 * r + 1);

  // A running window, deliberately, rather than the prefix sums this reads like
  // it wants: the CUDA path has to produce the same numbers, and the same
  // additions in the same order is the only way to promise that.
  for (int l = 0; l < lines; ++l) {
    const float* ip = in + lineStep * l;
    float* op = out + lineStep * l;

    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i <= r && i < n; ++i)
      for (int ch = 0; ch < 4; ++ch) sum[ch] += ip[step * i + ch];

    for (int i = 0; i < n; ++i) {
      // Divided by the FULL window even at the ends, where part of it hangs off
      // the tile. Outside there is nothing, and nothing is transparent -- so the
      // edge fades out instead of smearing the border pixel outwards.
      for (int ch = 0; ch < 4; ++ch) op[step * i + ch] = sum[ch] * inv;
      const int add = i + r + 1, drop = i - r;
      if (add < n)
        for (int ch = 0; ch < 4; ++ch) sum[ch] += ip[step * add + ch];
      if (drop >= 0)
        for (int ch = 0; ch < 4; ++ch) sum[ch] -= ip[step * drop + ch];
    }
  }
}

// `radius` is the visible spread in pixels, matching what the old disc jitter
// covered, so a project's Start Blur means roughly the same distance as before
// even though it now looks like a blur instead of a row of copies.
void blurTile(std::vector<float>& tile, std::vector<float>& scratch, int w, int h, float radius) {
  const int r = std::max(1, int(radius / 3.0f + 0.5f));
  if (w <= 0 || h <= 0) return;
  scratch.assign(tile.size(), 0.0f);
  for (int pass = 0; pass < 3; ++pass) {
    boxPass(tile.data(), scratch.data(), w, h, r, true);
    boxPass(scratch.data(), tile.data(), w, h, r, false);
  }
}

}  // namespace

void compositeGroups(const ImageView& dst, const ImageView& src, const Segmentation& seg,
                     const std::vector<GroupTransform>& transforms, int taps,
                     const RectI& renderWindow, const float* pivots) {
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
  std::vector<float> tile, scratch;  // reused across groups

  for (size_t gi = 0; gi < seg.groups.size(); ++gi) {
    const GroupTransform* tg = &transforms[gi * size_t(taps)];

    const RectI& b = seg.groups[gi].bbox;
    const float cx = pivots ? pivots[gi * 2 + 0] : 0.5f * float(b.x1 + b.x2);
    const float cy = pivots ? pivots[gi * 2 + 1] : 0.5f * float(b.y1 + b.y2);

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

    // Defocus, averaged over the live taps. It barely moves across one shutter,
    // and one radius for the group is what lets the blur be a single pass over
    // the finished sprite rather than something each tap carries.
    float radius = 0.0f;
    {
      int n = 0;
      for (int k = 0; k < taps; ++k) {
        const GroupTransform& t = geom[k].t;
        if (!t.visible || t.opacity <= 0.0f) continue;
        radius += t.blur;
        ++n;
      }
      if (n > 0) radius /= float(n);
    }

    // The group is drawn into its own tile first. Blurring wants the sprite
    // whole and by itself -- blurring straight into the frame would smear it
    // into whatever was composited under it.
    const int tw = dest.width(), th = dest.height();
    tile.assign(size_t(tw) * size_t(th) * 4, 0.0f);

    for (int y = 0; y < th; ++y) {
      float* out = &tile[size_t(y) * size_t(tw) * 4];
      for (int x = 0; x < tw; ++x, out += 4) {
        const float px = float(dest.x1 + x), py = float(dest.y1 + y);
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int k = 0; k < taps; ++k) {
          const GroupTransform& t = geom[k].t;
          if (!t.visible || t.opacity <= 0.0f) continue;

          float fx, fy;
          geom[k].inverse(px, py, &fx, &fy);

          float smp[4];
          sampleMasked(src, seg, int(gi), fx, fy, smp);
          if (smp[3] <= 0.0f) continue;

          acc[0] += smp[0] * t.opacity;
          acc[1] += smp[1] * t.opacity;
          acc[2] += smp[2] * t.opacity;
          acc[3] += smp[3] * t.opacity;
        }

        out[0] = acc[0] * invTaps;
        out[1] = acc[1] * invTaps;
        out[2] = acc[2] * invTaps;
        out[3] = acc[3] * invTaps;
      }
    }

    if (radius > 0.5f) blurTile(tile, scratch, tw, th, radius);

    for (int y = 0; y < th; ++y) {
      const float* in = &tile[size_t(y) * size_t(tw) * 4];
      float* out = dst.at(dest.x1, dest.y1 + y);
      for (int x = 0; x < tw; ++x, in += 4, out += 4) {
        const float a = in[3];
        if (a <= 0.0f) continue;
        const float ia = 1.0f - a;
        out[0] = in[0] + out[0] * ia;
        out[1] = in[1] + out[1] * ia;
        out[2] = in[2] + out[2] * ia;
        out[3] = a + out[3] * ia;
      }
    }
  }
}

}  // namespace rta
