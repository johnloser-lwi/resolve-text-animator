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
  for (size_t gi = 0; gi < seg.groups.size(); ++gi) {
    const RectI& b = seg.groups[gi].bbox;
    const float* c = kPalette[gi % 8];
    for (int k = 0; k < t; ++k) {
      for (int x = b.x1 - k; x < b.x2 + k; ++x) {
        plot(dst, x, b.y1 - k, c);
        plot(dst, x, b.y2 + k - 1, c);
      }
      for (int y = b.y1 - k; y < b.y2 + k; ++y) {
        plot(dst, b.x1 - k, y, c);
        plot(dst, b.x2 + k - 1, y, c);
      }
    }
  }
}

}  // namespace rta
