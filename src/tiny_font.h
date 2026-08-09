// A 3x5 bitmap digit font.
//
// Show Detection numbers each group so a manual merge or split can name one, and
// the number has to appear in the rendered image rather than the viewer overlay:
// the overlay is a separate surface the CUDA path never touches, and the two
// paths must number the groups identically or an override would silently
// retarget.
//
// Header-only, and every entry point is callable from device code, so the CPU
// and CUDA diagnostics draw from one definition.
#pragma once

#if defined(__CUDACC__)
#define RTA_FONT_HD __host__ __device__
#else
#define RTA_FONT_HD
#endif

namespace rta {

// One byte per column, low bit is the top row; three columns per digit.
//
// Held inside the function rather than at namespace scope because device code
// cannot reach a host constant, and duplicating the table into __constant__
// memory would let the two copies drift.
RTA_FONT_HD inline unsigned char digitColumn(int digit, int col) {
  const unsigned char table[30] = {
      0x1F, 0x11, 0x1F,  // 0
      0x00, 0x1F, 0x00,  // 1
      0x1D, 0x15, 0x17,  // 2
      0x15, 0x15, 0x1F,  // 3
      0x07, 0x04, 0x1F,  // 4
      0x17, 0x15, 0x1D,  // 5
      0x1F, 0x15, 0x1D,  // 6
      0x01, 0x01, 0x1F,  // 7
      0x1F, 0x15, 0x1F,  // 8
      0x17, 0x15, 0x1F,  // 9
  };
  if (digit < 0 || digit > 9 || col < 0 || col > 2) return 0;
  return table[digit * 3 + col];
}

// True when (col,row) of `digit` is lit, at an integer `scale`.
RTA_FONT_HD inline bool digitPixel(int digit, int col, int row, int scale) {
  if (scale < 1) return false;
  const int c = col / scale, r = row / scale;
  if (c < 0 || c > 2 || r < 0 || r > 4) return false;
  return (digitColumn(digit, c) >> r) & 1;
}

// Digits needed to write `value`.
RTA_FONT_HD inline int digitCount(int value) {
  int n = 1;
  for (int v = value; v >= 10; v /= 10) ++n;
  return n;
}

// Width in pixels of `value` drawn at `scale`, digits separated by one column.
RTA_FONT_HD inline int digitsWidth(int value, int scale) {
  const int n = digitCount(value);
  return n * (3 * scale) + (n - 1) * scale;
}

}  // namespace rta
