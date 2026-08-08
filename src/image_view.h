// Host-independent view over a premultiplied float RGBA buffer.
// Everything outside plugin.cpp works in terms of this, which is what lets the
// same segmentation code run inside Resolve and inside tools/segtest.
#pragma once

#include <cstddef>
#include <cstdint>

namespace rta {

struct RectI {
  int x1 = 0, y1 = 0, x2 = 0, y2 = 0;  // [x1,x2) [y1,y2)

  int width() const { return x2 - x1; }
  int height() const { return y2 - y1; }
  bool empty() const { return x2 <= x1 || y2 <= y1; }

  void grow(int px) {
    x1 -= px;
    y1 -= px;
    x2 += px;
    y2 += px;
  }

  void unionWith(const RectI& o) {
    if (o.empty()) return;
    if (empty()) {
      *this = o;
      return;
    }
    if (o.x1 < x1) x1 = o.x1;
    if (o.y1 < y1) y1 = o.y1;
    if (o.x2 > x2) x2 = o.x2;
    if (o.y2 > y2) y2 = o.y2;
  }
};

// A non-owning view. `data` points at the pixel whose coordinate is (0,0);
// rowStride is in floats, not bytes, and may be negative for bottom-up images.
struct ImageView {
  float* data = nullptr;
  int width = 0;
  int height = 0;
  std::ptrdiff_t rowStride = 0;  // in floats

  float* row(int y) const { return data + rowStride * y; }
  float* at(int x, int y) const { return row(y) + std::ptrdiff_t(x) * 4; }
  bool valid() const { return data != nullptr && width > 0 && height > 0; }
};

}  // namespace rta
