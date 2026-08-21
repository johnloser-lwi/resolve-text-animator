// Geometry shared by the CPU and CUDA compositors, so the two paths cannot
// drift apart on how a group's transform is defined.
//
// Forward map:  p' = centre + R(theta) * scale * (p - centre) + offset
// Rendering walks destination pixels, so the inner loop needs the inverse.
#pragma once

#include <algorithm>
#include <cmath>

#include "animator.h"
#include "image_view.h"

namespace rta {

// Clamp to +-2^24 so float -> int conversion is always defined. Past that a
// coordinate is millions of pixels outside any frame, so nothing it could have
// sampled changes. NaN fails both comparisons and lands on the far side, where
// every edge test fails -- transparent, the only sane answer to NaN.
inline float boundCoord(float v) {
  const float lim = 16777216.0f;
  if (v < lim) return v > -lim ? v : -lim;
  return lim;  // v >= lim, or NaN
}

struct TapGeom {
  GroupTransform t;
  float cx = 0.0f, cy = 0.0f;
  float m00 = 1.0f, m01 = 0.0f, m10 = 0.0f, m11 = 1.0f;

  TapGeom() = default;
  TapGeom(const GroupTransform& tr, float centreX, float centreY)
      : t(tr), cx(centreX), cy(centreY) {
    const float inv = 1.0f / (std::fabs(tr.scale) < 1e-6f ? 1e-6f : tr.scale);
    const float c = std::cos(-tr.rotation) * inv;
    const float s = std::sin(-tr.rotation) * inv;
    m00 = c;
    m01 = -s;
    m10 = s;
    m11 = c;
  }

  void inverse(float x, float y, float* fx, float* fy) const {
    const float dx = x + 0.5f - cx - t.offsetX;
    const float dy = y + 0.5f - cy - t.offsetY;
    *fx = m00 * dx + m01 * dy + cx - 0.5f;
    *fy = m10 * dx + m11 * dy + cy - 0.5f;
  }
};

// Axis-aligned bounds of a group's box under one tap's forward transform.
// Once rotation is involved the corners must be mapped individually.
inline RectI tapBounds(const GroupTransform& t, const RectI& b, float cx, float cy) {
  const float c = std::cos(t.rotation) * t.scale;
  const float s = std::sin(t.rotation) * t.scale;
  const float xs[4] = {float(b.x1), float(b.x2), float(b.x1), float(b.x2)};
  const float ys[4] = {float(b.y1), float(b.y1), float(b.y2), float(b.y2)};

  float lox = 1e30f, loy = 1e30f, hix = -1e30f, hiy = -1e30f;
  for (int i = 0; i < 4; ++i) {
    const float dx = xs[i] - cx, dy = ys[i] - cy;
    const float px = cx + c * dx - s * dy + t.offsetX;
    const float py = cy + s * dx + c * dy + t.offsetY;
    lox = std::min(lox, px);
    hix = std::max(hix, px);
    loy = std::min(loy, py);
    hiy = std::max(hiy, py);
  }
  // Padding must never be negative: this rect is the region the compositor is
  // allowed to write, so a negative pad silently crops the group rather than
  // making room for its blur.
  const float pad = std::max(0.0f, t.blur);
  // Bounded for the same reason the sampler bounds: this rect is grown and
  // intersected next, and an INT_MIN from an undefined conversion wraps on the
  // first grow.
  return RectI{int(std::floor(boundCoord(lox - pad))), int(std::floor(boundCoord(loy - pad))),
               int(std::ceil(boundCoord(hix + pad))), int(std::ceil(boundCoord(hiy + pad)))};
}

}  // namespace rta
