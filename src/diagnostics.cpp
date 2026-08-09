#include "diagnostics.h"

#include <algorithm>

namespace rta {
namespace {

// Distinct, high-contrast hues so adjacent groups never read as one.
const float kPalette[8][3] = {
    {1.0f, 0.20f, 0.25f}, {0.20f, 0.85f, 1.0f}, {1.0f, 0.85f, 0.15f},
    {0.35f, 1.0f, 0.35f}, {1.0f, 0.45f, 1.0f},  {0.55f, 0.55f, 1.0f},
    {1.0f, 0.60f, 0.20f}, {0.20f, 1.0f, 0.75f},
};

void plot(const ImageView& dst, int x, int y, const float* c) {
  if (x < 0 || y < 0 || x >= dst.width || y >= dst.height) return;
  float* p = dst.at(x, y);
  p[0] = c[0];
  p[1] = c[1];
  p[2] = c[2];
  p[3] = 1.0f;
}

}  // namespace

void drawDiagnostics(const ImageView& dst, const Segmentation& seg, int lineWidth) {
  if (!dst.valid()) return;
  const int t = std::max(1, lineWidth);
  const int yMid = seg.height / 2;

  for (size_t gi = 0; gi < seg.groups.size(); ++gi) {
    const Group& g = seg.groups[gi];
    const RectI& b = g.bbox;
    const float* c = kPalette[gi % 8];

    // The outline leans with the type. Sides follow the slant, top and bottom
    // stay horizontal, so the box traces the parallelogram the word actually
    // occupies -- an upright rectangle round an italic word runs through the
    // letters either side of it and reads as though detection had merged them.
    //
    // Degenerate extents (a group whose deskewed range never got set) fall back
    // to the axis-aligned box rather than collapsing to a line.
    const bool sheared = g.sx2 > g.sx1;
    for (int k = 0; k < t; ++k) {
      // Sides.
      for (int y = b.y1 - k; y < b.y2 + k; ++y) {
        const float shear = sheared ? float(y - yMid) * seg.slantTan : 0.0f;
        const int left = sheared ? int(g.sx1 + shear) : b.x1;
        const int right = sheared ? int(g.sx2 + shear) : b.x2;
        plot(dst, left - k, y, c);
        plot(dst, right + k - 1, y, c);
      }
      // Top and bottom, each at its own row's shear.
      const int rows[2] = {b.y1 - k, b.y2 + k - 1};
      for (int r = 0; r < 2; ++r) {
        const int y = rows[r];
        const float shear = sheared ? float(y - yMid) * seg.slantTan : 0.0f;
        const int left = sheared ? int(g.sx1 + shear) : b.x1;
        const int right = sheared ? int(g.sx2 + shear) : b.x2;
        for (int x = left - k; x < right + k; ++x) plot(dst, x, y, c);
      }
    }
  }
}

}  // namespace rta
