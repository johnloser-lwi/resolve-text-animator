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
  return RectI{int(std::floor(lox - pad)), int(std::floor(loy - pad)),
               int(std::ceil(hix + pad)), int(std::ceil(hiy + pad))};
}

}  // namespace rta
