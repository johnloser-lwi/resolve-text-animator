#include "segmentation.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <string>

namespace rta {
namespace {

// ---------------------------------------------------------------- union-find

struct UnionFind {
  std::vector<int> parent;

  int add() {
    parent.push_back(int(parent.size()));
    return int(parent.size()) - 1;
  }

  int find(int a) {
    while (parent[a] != a) {
      parent[a] = parent[parent[a]];  // path halving
      a = parent[a];
    }
    return a;
  }

  void unite(int a, int b) {
    a = find(a);
    b = find(b);
    if (a != b) parent[std::max(a, b)] = std::min(a, b);
  }
};

// ------------------------------------------------------------------ dilation

// Separable binary dilation. Used to bridge the hairline joins of script fonts
// so that a cursive word labels as one component instead of shattering.
//
// Cost does not depend on the radius. The obvious version scans the window at
// every pixel, so it slows down as the radius grows -- which is felt exactly
// when dragging the Bridge Radius slider, since every change re-runs the whole
// analysis. Instead each line is swept twice, carrying the position of the
// nearest set pixel behind and ahead; a pixel is set if either is within r.
// Exact for a binary mask, and identical to the window scan it replaces.
void dilateLine(const uint8_t* src, uint8_t* dst, int n, std::ptrdiff_t stride, int r,
                std::vector<int>& prev, std::vector<int>& next) {
  const int kFar = std::numeric_limits<int>::max() / 4;

  int last = -kFar;
  for (int i = 0; i < n; ++i) {
    if (src[i * stride]) last = i;
    prev[size_t(i)] = last;
  }
  int ahead = kFar;
  for (int i = n - 1; i >= 0; --i) {
    if (src[i * stride]) ahead = i;
    next[size_t(i)] = ahead;
  }
  for (int i = 0; i < n; ++i)
    dst[i * stride] = uint8_t((i - prev[size_t(i)] <= r) || (next[size_t(i)] - i <= r));
}

std::vector<uint8_t> dilate(const std::vector<uint8_t>& src, int w, int h, int r) {
  std::vector<uint8_t> tmp(size_t(w) * h, 0);
  std::vector<uint8_t> out(size_t(w) * h, 0);
  std::vector<int> prev(size_t(std::max(w, h))), next(size_t(std::max(w, h)));

  for (int y = 0; y < h; ++y)
    dilateLine(&src[size_t(y) * w], &tmp[size_t(y) * w], w, 1, r, prev, next);
  for (int x = 0; x < w; ++x)
    dilateLine(&tmp[size_t(x)], &out[size_t(x)], h, w, r, prev, next);
  return out;
}

// ----------------------------------------------------------------- component

struct Component {
  RectI bbox;
  // Horizontal extent measured in the deskewed frame. For upright text this is
  // just bbox.x1/x2; for an italic it is the range the letter would occupy if
  // it were standing straight, which is what grouping has to reason about.
  float sx1 = 0.0f, sx2 = 0.0f;
  int area = 0;
  bool alive = false;
};

// Estimates the italic slant of the text, as tan(angle).
//
// Sheared by the right amount, the vertical stems of every letter line up into
// tall narrow columns, so the vertical projection profile becomes as peaky as it
// can get. Sum of squares measures exactly that peakiness -- it is maximised
// when the same ink is concentrated into fewer columns -- so the best shear is
// the one that scores highest. This is the classic deskew, and it needs no
// knowledge of the typeface.
float detectSlant(const std::vector<uint8_t>& mask, int w, int h) {
  // One pass to collect the ink, subsampled: the estimate only needs the
  // overall distribution, and this keeps the search over candidates cheap.
  // Rows are subsampled, columns are NOT. Skipping columns quietly rigs the
  // measurement: with a stride of 2 every sample lands on an even column at zero
  // shear, so all the ink falls in half the bins and the score doubles, while any
  // real shear scatters samples across both parities and scores lower. Zero then
  // wins every time and a 20-degree italic reads as upright.
  //
  // The row stride is a FRACTION OF FRAME HEIGHT, not a fixed 3. Fixed, it takes
  // half as many samples from a half-resolution render of the same title, so the
  // estimate quietly weakens in proxy and can fall under the floor below and
  // report upright -- which regroups the whole line. Scaling the stride keeps the
  // sample density constant at any resolution, and lands on exactly 3 at 1080p,
  // where it was tuned.
  const int rowStride = std::max(1, h / 360);
  std::vector<int> px, py;
  px.reserve(size_t(w) * h / 16);
  for (int y = 0; y < h; y += rowStride) {
    const uint8_t* row = &mask[size_t(y) * w];
    for (int x = 0; x < w; ++x) {
      if (!row[x]) continue;
      px.push_back(x);
      py.push_back(y);
    }
  }
  // Absolute on purpose, and stable now that the density is: this asks whether
  // there is enough DATA to fit a slant, which is a question about sample count,
  // not about scale.
  if (px.size() < 64) return 0.0f;  // too little ink to say anything

  const int yMid = h / 2;
  std::vector<int> hist(size_t(w) + 1, 0);

  auto score = [&](float t) {
    std::fill(hist.begin(), hist.end(), 0);
    for (size_t i = 0; i < px.size(); ++i) {
      // floor, not truncation: int() rounds toward zero and so biases rows above
      // the middle differently from rows below, blurring the very peak we want.
      const int sx = px[i] - int(std::floor(float(py[i] - yMid) * t));
      if (sx < 0 || sx > w) continue;
      ++hist[size_t(sx)];
    }
    double s = 0.0;
    for (int c : hist) s += double(c) * double(c);
    return s;
  };

  // Coarse sweep. +-0.7 covers roughly +-35 degrees, past any real oblique face.
  double bestScore = score(0.0f);
  float bestTan = 0.0f;  // upright unless something clearly beats it
  for (int step = -35; step <= 35; ++step) {
    const float t = float(step) * 0.02f;
    const double s = score(t);
    // Must beat the incumbent by a margin, so upright text is never nudged off
    // zero by noise in the profile.
    if (s > bestScore * 1.002) {
      bestScore = s;
      bestTan = t;
    }
  }

  // Refine around the winner: the coarse step is about 1.1 degrees, enough to
  // leave a couple of degrees of error on the estimate.
  const float coarse = bestTan;
  for (int step = -4; step <= 4; ++step) {
    const float t = coarse + float(step) * 0.005f;
    const double s = score(t);
    if (s > bestScore) {
      bestScore = s;
      bestTan = t;
    }
  }
  return bestTan;
}

// Fraction of the narrower span that the two spans share in x.
float xOverlapRatio(float a1, float a2, float b1, float b2) {
  const float overlap = std::min(a2, b2) - std::max(a1, b1);
  if (overlap <= 0.0f) return 0.0f;
  return overlap / std::max(1.0f, std::min(a2 - a1, b2 - b1));
}

// Fraction of the shorter box's height that the two boxes share in y.
float yOverlapRatio(const RectI& a, const RectI& b) {
  const int overlap = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);
  if (overlap <= 0) return 0.0f;
  return float(overlap) / float(std::max(1, std::min(a.height(), b.height())));
}

// True when two components are parts of one glyph rather than two letters.
//
// Horizontal overlap alone is not enough: in a slanted or script face every
// letter overlaps its neighbour in x, which would chain a whole line into one
// glyph. What actually distinguishes a diacritic is that it sits clear above
// (or below) its base with no vertical overlap, whereas adjacent letters
// always share most of their vertical extent.
bool sameGlyph(const Component& a, const Component& b) {
  return xOverlapRatio(a.sx1, a.sx2, b.sx1, b.sx2) >= 0.6f &&
         yOverlapRatio(a.bbox, b.bbox) <= 0.2f;
}

// ------------------------------------------------------------ reference text

// What a character is allowed to be made of: how many marks, and in what
// arrangement.
//
// This is what makes the string worth more than its length. Knowing that
// character 27 is a quote says the two strokes there belong together; knowing
// only that the line holds 28 characters does not, and gap size alone gets it
// wrong -- an 'r' kerns tighter against a following quote than the quote's own
// two strokes are spaced, so "merge the closest pair" fuses the r to half the
// quote and leaves the other half stranded.
//
// The SHAPE matters as much as the count, and leaving it out was a bug worth
// recording. Allowing 'i' two marks unconditionally is true -- a dotted i is a
// dot and a stem -- but §7 has almost always joined those already, so the spare
// allowance had nothing of its own left to claim and was spent swallowing the
// NEXT LETTER instead, at no cost. Which merge won then came down to gap size
// again, and a trailing full stop was enough to tip it: an 8px full stop drags
// down the median glyph height that normalises gaps nearby, the closing quote's
// own gap normalises larger, and the bogus merge became the cheaper reading.
//
// So each character says how its marks must SIT relative to one another, and a
// run that does not look like that is not the character, whatever the count.
enum class MarkShape {
  Single,       // one mark; nothing to check
  SideBySide,   // strokes in the same vertical band -- a quote, an ellipsis
  Stacked,      // one part clear above the other -- a dot, a diacritic, a colon
};

struct CharSpec {
  int allow = 1;
  MarkShape shape = MarkShape::Single;
};

CharSpec expectedMarks(uint32_t cp) {
  switch (cp) {
    // Side by side. These are the ones §7 CANNOT join: the strokes share a
    // vertical band, so the diacritic rule's demand for vertical separation
    // never fires, and no amount of tuning it would help.
    case 0x0022:  // "
    case 0x201C:  // “
    case 0x201D:  // ”
    case 0x201E:  // „
    case 0x201F:
    case 0x2033:  // ″
    case 0x00AB:  // «
    case 0x00BB:  // »
    case 0x2039:  // ‹
    case 0x203A:  // ›
      return {2, MarkShape::SideBySide};
    case 0x2034:  // ‴
    case 0x2026:  // …
    case 0x0025:  // %
      return {3, MarkShape::SideBySide};
    case 0x2030:  // ‰
      return {4, MarkShape::SideBySide};

    // Stacked. §7 normally joins these already, so the allowance is a safety net
    // for a face where it misses -- and, because the shape is checked, a net that
    // cannot be spent on a neighbouring letter.
    case 0x00A8:  // ¨
    case 0x003D:  // =
    case 0x003A:  // :
    case 0x003B:  // ;
    case 0x0021:  // !
    case 0x003F:  // ?
    case 0x0069:  // i
    case 0x006A:  // j
      return {2, MarkShape::Stacked};

    default:
      // Accented Latin and beyond: the mark may or may not have been folded into
      // its base already, so allow the pair without insisting on it.
      return cp >= 0x00C0u ? CharSpec{2, MarkShape::Stacked} : CharSpec{1, MarkShape::Single};
  }
}

// Do two marks sit the way this character's parts have to sit?
//
// SideBySide wants the same vertical band AND similar heights. The band alone is
// not enough: a quote stroke lies entirely within the vertical span of the letter
// beside it, so it "overlaps" that letter completely -- it is the height that
// tells a 23px stroke from the 66px 't' next to it.
//
// Stacked wants them vertically clear of each other, which is the same test §7
// uses for a dot and its stem.
bool shapeFits(MarkShape s, const RectI& a, const RectI& b) {
  const float yo = yOverlapRatio(a, b);
  if (s == MarkShape::Stacked) return yo <= 0.3f;
  const float ha = float(std::max(1, a.height())), hb = float(std::max(1, b.height()));
  return yo >= 0.5f && std::min(ha, hb) / std::max(ha, hb) >= 0.6f;
}

// The reference string as words, each word a list of PER-CHARACTER specs -- so a
// word's length is its character count and its contents say which of those
// characters may legitimately be several marks, and how those marks must sit.
//
// Read in CODEPOINTS, not bytes: a two-byte 'a-umlaut' is one letter on screen
// and has to be one here. A combining mark is not a character of its own --
// glyph assembly folds a diacritic into its base -- so it does not add to the
// count; it raises the allowance of the character it modifies instead, which is
// what a decomposed 'a' + U+0308 actually costs in marks. Punctuation is an
// ordinary character throughout: a full stop counts here exactly as it counts in
// the picture, which is what keeps the two sides of the sum honest.
std::vector<std::vector<CharSpec>> referenceWords(const std::string& s) {
  std::vector<std::vector<CharSpec>> words;
  std::vector<CharSpec> cur;
  for (size_t i = 0; i < s.size();) {
    const unsigned char b = uint8_t(s[i]);
    // Everything below the space is a separator -- spaces, tabs, newlines. UTF-8
    // never puts a byte this low inside a multi-byte sequence, so this cannot
    // cut one in half.
    if (b <= ' ') {
      if (!cur.empty()) words.push_back(std::move(cur));
      cur.clear();
      ++i;
      continue;
    }

    uint32_t cp = b;
    size_t len = 1;
    if ((b & 0xE0u) == 0xC0u) {
      cp = b & 0x1Fu;
      len = 2;
    } else if ((b & 0xF0u) == 0xE0u) {
      cp = b & 0x0Fu;
      len = 3;
    } else if ((b & 0xF8u) == 0xF0u) {
      cp = b & 0x07u;
      len = 4;
    }
    for (size_t k = 1; k < len && i + k < s.size(); ++k)
      cp = (cp << 6) | (uint8_t(s[i + k]) & 0x3Fu);
    i += len;

    if (cp >= 0x0300u && cp <= 0x036Fu) {  // combining mark: part of its base
      if (!cur.empty()) {
        ++cur.back().allow;
        cur.back().shape = MarkShape::Stacked;  // a diacritic sits above its base
      }
      continue;
    }
    cur.push_back(expectedMarks(cp));
  }
  if (!cur.empty()) words.push_back(std::move(cur));
  return words;
}

}  // namespace

// ---------------------------------------------------------------------------

uint64_t hashAlpha(const ImageView& src, float threshold) {
  uint64_t h = 1469598103934665603ull;  // FNV-1a offset basis
  if (!src.valid()) return h;
  for (int y = 0; y < src.height; y += 8) {
    const float* row = src.row(y);
    for (int x = 0; x < src.width; x += 8) {
      // Quantise to 16 levels so imperceptible float noise doesn't invalidate
      // the cache, while a genuine text change always does.
      const uint8_t q = uint8_t(std::min(15.0f, std::max(0.0f, row[x * 4 + 3]) * 15.0f));
      h = (h ^ uint64_t(q + (row[x * 4 + 3] > threshold ? 16 : 0))) * 1099511628211ull;
    }
  }
  h = (h ^ uint64_t(src.width)) * 1099511628211ull;
  h = (h ^ uint64_t(src.height)) * 1099511628211ull;
  return h;
}

Segmentation segment(const ImageView& src, const DetectParams& params) {
  Segmentation seg;
  if (!src.valid()) return seg;

  const int w = src.width, h = src.height;
  seg.width = w;
  seg.height = h;

  // 1. Binary mask from alpha.
  std::vector<uint8_t> mask(size_t(w) * h, 0);
  bool any = false;
  for (int y = 0; y < h; ++y) {
    const float* row = src.row(y);
    uint8_t* m = &mask[size_t(y) * w];
    for (int x = 0; x < w; ++x) {
      if (row[x * 4 + 3] > params.alphaThreshold) {
        m[x] = 1;
        any = true;
      }
    }
  }
  if (!any) return seg;

  // 1b. Letter height, in pixels, for the relative controls.
  //
  // Taken from the INK PROFILE rather than from glyphs, because bridging happens
  // next and needs the answer before anything has been labelled. Rows carrying
  // ink form bands; the median band height is the line height, which tracks proxy
  // scale, timeline resolution and point size exactly as letter height does.
  //
  // Median over bands, so a stray one-line caption beside a large title does not
  // set the scale for both.
  float letterHeight = 0.0f;
  {
    std::vector<uint8_t> rowInk(size_t(h), 0);
    for (int y = 0; y < h; ++y) {
      const uint8_t* row = &mask[size_t(y) * w];
      for (int x = 0; x < w; ++x) {
        if (row[x]) {
          rowInk[size_t(y)] = 1;
          break;
        }
      }
    }
    std::vector<int> bands;
    for (int y = 0; y < h;) {
      if (!rowInk[size_t(y)]) {
        ++y;
        continue;
      }
      const int start = y;
      while (y < h && rowInk[size_t(y)]) ++y;
      bands.push_back(y - start);
    }
    if (!bands.empty()) {
      std::nth_element(bands.begin(), bands.begin() + bands.size() / 2, bands.end());
      letterHeight = float(bands[bands.size() / 2]);
    }
  }

  seg.letterHeight = letterHeight;

  // Relative settings resolved to pixels, ADDED to the legacy pixel ones so an
  // older project keeps the grouping it was saved with.
  const int bridgePx =
      params.bridgeRadius + int(std::lround(double(params.charPadding) * double(letterHeight)));
  // At least 1: a despeckle that discards nothing still has to discard nothing,
  // not everything, if the frame turns out to hold no measurable type.
  const int minArea = std::max(
      1,
      int(std::lround(double(params.minMarkArea) * double(letterHeight) * double(letterHeight))));

  // 2. Optional bridge dilation. Labels are computed on the dilated mask but
  //    written back only onto true text pixels, so bboxes stay accurate.
  std::vector<uint8_t> dilated;
  if (bridgePx > 0) dilated = dilate(mask, w, h, bridgePx);
  const std::vector<uint8_t>& labelMask = bridgePx > 0 ? dilated : mask;

  // 3. Connected components, 8-connectivity, two-pass with union-find.
  std::vector<int32_t> prov(size_t(w) * h, 0);
  UnionFind uf;
  uf.add();  // label 0 == background
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const size_t p = size_t(y) * w + x;
      if (!labelMask[p]) continue;
      int best = 0;
      auto consider = [&](int nx, int ny) {
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) return;
        const int32_t l = prov[size_t(ny) * w + nx];
        if (l == 0) return;
        if (best == 0) {
          best = l;
        } else {
          uf.unite(best, l);
          best = std::min(uf.find(best), uf.find(l));
        }
      };
      consider(x - 1, y);
      consider(x - 1, y - 1);
      consider(x, y - 1);
      consider(x + 1, y - 1);
      prov[p] = best != 0 ? best : uf.add();
    }
  }
  // Compact roots to sequential labels.
  std::vector<int> remap(uf.parent.size(), 0);
  int labelCount = 0;
  for (size_t i = 1; i < uf.parent.size(); ++i) {
    if (uf.find(int(i)) == int(i)) remap[i] = ++labelCount;
  }
  if (labelCount == 0) return seg;

  // 4. Write labels back onto true text pixels only, and gather stats.
  seg.labelImage.assign(size_t(w) * h, 0);
  // Italic compensation, measured once for the whole image: the slant is a
  // property of the typeface, so estimating it per line would only add noise.
  // Sign convention for the outside world: POSITIVE degrees means the letters
  // lean right, the way a normal italic does. Internally the shear that undoes
  // that lean is the negative of it, which is what slantTan holds.
  constexpr float kPi = 3.14159265358979323846f;
  // Objects stand upright by definition: there is no baseline to lean off, and a
  // slant measured from a drawing is noise that would only scramble the order
  // things are read in.
  const bool typeMode = params.mode != GroupMode::Object;
  const float slantTan = !typeMode              ? 0.0f
                         : params.autoSlant     ? detectSlant(mask, w, h)
                                                : -std::tan(params.italicSlant * kPi / 180.0f);
  const int yMid = h / 2;
  seg.slantDegrees = -std::atan(slantTan) * 180.0f / kPi;
  seg.slantTan = slantTan;

  std::vector<Component> comps(size_t(labelCount) + 1);
  for (int y = 0; y < h; ++y) {
    // Deskewed x of this row. Only the measurement shears; seg.labelImage still
    // records where the pixels really are, so rendering is untouched.
    const float shear = float(y - yMid) * slantTan;
    for (int x = 0; x < w; ++x) {
      const size_t p = size_t(y) * w + x;
      if (!mask[p] || prov[p] == 0) continue;
      const int lab = remap[uf.find(prov[p])];
      seg.labelImage[p] = lab;
      Component& c = comps[lab];
      const float sx = float(x) - shear;
      if (c.area == 0) {
        c.bbox = RectI{x, y, x + 1, y + 1};
        c.sx1 = sx;
        c.sx2 = sx + 1.0f;
      } else {
        c.bbox.x1 = std::min(c.bbox.x1, x);
        c.bbox.y1 = std::min(c.bbox.y1, y);
        c.bbox.x2 = std::max(c.bbox.x2, x + 1);
        c.bbox.y2 = std::max(c.bbox.y2, y + 1);
        c.sx1 = std::min(c.sx1, sx);
        c.sx2 = std::max(c.sx2, sx + 1.0f);
      }
      ++c.area;
    }
  }

  // 5. Despeckle: antialiasing debris and stray dots are not glyphs.
  for (int l = 1; l <= labelCount; ++l) comps[l].alive = comps[l].area >= minArea;
  for (size_t p = 0; p < seg.labelImage.size(); ++p) {
    const int32_t l = seg.labelImage[p];
    if (l != 0 && !comps[l].alive) seg.labelImage[p] = 0;
  }

  // 6. Lines by horizontal projection profile: rows with no text separate lines.
  std::vector<int> rowCount(h, 0);
  for (int y = 0; y < h; ++y) {
    const int32_t* r = &seg.labelImage[size_t(y) * w];
    int n = 0;
    for (int x = 0; x < w; ++x) n += (r[x] != 0);
    rowCount[y] = n;
  }
  std::vector<RectI> lineSpans;  // only y1/y2 meaningful here
  for (int y = 0; y < h;) {
    if (rowCount[y] == 0) {
      ++y;
      continue;
    }
    const int start = y;
    while (y < h && rowCount[y] != 0) ++y;
    lineSpans.push_back(RectI{0, start, w, y});
  }

  // 6b. A band too thin to be a line is a MARK with clear rows either side of
  //     it: the dot of an i on a line with no ascenders, where nothing reaches
  //     up to bridge the gap between dot and stem. Left alone it becomes a line
  //     of its own, and then nothing can rejoin it to the stem -- glyph
  //     assembly works within a line -- so the reveal gets a ten-pixel unit
  //     numbered wherever that phantom line falls in reading order.
  //
  // Only a line with NO ascender triggers it, which is why it looked arbitrary
  // from outside: "easy views" broke and "easy vibes" would not have, because
  // the b fills the rows the dot sits in.
  //
  // Merged only when the thin band sits directly over or under something in
  // the neighbouring band that it overlaps in x, within a short gap -- the
  // geometry of a mark and its base -- so a genuine row of small shapes with
  // nothing above or below it is left as the line it is.
  if (lineSpans.size() > 1) {
    int refH = 0;
    for (const RectI& sp : lineSpans) refH = std::max(refH, sp.height());
    const int thinH = int(0.3f * float(refH));
    const int maxGap = int(0.5f * float(refH));

    auto bandOf = [&](int l) {
      const int cy = (comps[l].bbox.y1 + comps[l].bbox.y2) / 2;
      for (size_t i = 0; i < lineSpans.size(); ++i)
        if (cy >= lineSpans[i].y1 && cy < lineSpans[i].y2) return int(i);
      return -1;
    };

    bool merged = true;
    while (merged && lineSpans.size() > 1) {
      merged = false;
      for (size_t i = 0; i < lineSpans.size(); ++i) {
        if (lineSpans[i].height() >= thinH) continue;

        int pick = -1;
        int pickGap = INT_MAX;
        for (int dir = -1; dir <= 1; dir += 2) {
          const int j = int(i) + dir;
          if (j < 0 || size_t(j) >= lineSpans.size()) continue;
          const int gap = dir < 0 ? lineSpans[i].y1 - lineSpans[size_t(j)].y2
                                  : lineSpans[size_t(j)].y1 - lineSpans[i].y2;
          if (gap > maxGap) continue;

          // Something in the neighbour directly under (or over) this band's ink.
          bool over = false;
          for (int a = 1; a <= labelCount && !over; ++a) {
            if (!comps[a].alive || bandOf(a) != int(i)) continue;
            for (int b = 1; b <= labelCount; ++b) {
              if (!comps[b].alive || bandOf(b) != j) continue;
              if (xOverlapRatio(comps[a].sx1, comps[a].sx2, comps[b].sx1, comps[b].sx2) >= 0.5f) {
                over = true;
                break;
              }
            }
          }
          if (over && gap < pickGap) {
            pick = j;
            pickGap = gap;
          }
        }
        if (pick < 0) continue;

        const size_t lo = size_t(std::min(int(i), pick)), hi = size_t(std::max(int(i), pick));
        lineSpans[lo].y1 = std::min(lineSpans[lo].y1, lineSpans[hi].y1);
        lineSpans[lo].y2 = std::max(lineSpans[lo].y2, lineSpans[hi].y2);
        lineSpans.erase(lineSpans.begin() + std::ptrdiff_t(hi));
        merged = true;
        break;
      }
    }
  }

  if (lineSpans.empty()) return seg;
  seg.lineCount = int(lineSpans.size());

  // Assign components to lines by bbox centre-y.
  std::vector<std::vector<int>> lineComps(lineSpans.size());
  for (int l = 1; l <= labelCount; ++l) {
    if (!comps[l].alive) continue;
    const int cy = (comps[l].bbox.y1 + comps[l].bbox.y2) / 2;
    size_t best = 0;
    int bestDist = INT32_MAX;
    for (size_t i = 0; i < lineSpans.size(); ++i) {
      if (cy >= lineSpans[i].y1 && cy < lineSpans[i].y2) {
        best = i;
        bestDist = 0;
        break;
      }
      const int d = cy < lineSpans[i].y1 ? lineSpans[i].y1 - cy : cy - lineSpans[i].y2 + 1;
      if (d < bestDist) {
        bestDist = d;
        best = i;
      }
    }
    lineComps[best].push_back(l);
  }

  // 7. Glyph assembly: components that substantially overlap in x are parts of
  //    one glyph -- the dot of an i/j, the bars of an =, an umlaut and its base.
  struct Glyph {
    RectI bbox;
    float sx1 = 0.0f, sx2 = 0.0f;  // deskewed extent, for ordering and gaps
    std::vector<int> labels;
  };
  std::vector<std::vector<Glyph>> lineGlyphs(lineSpans.size());
  for (size_t li = 0; li < lineComps.size(); ++li) {
    auto& ids = lineComps[li];
    if (ids.empty()) continue;

    UnionFind guf;
    for (size_t i = 0; i < ids.size(); ++i) guf.add();
    // Object mode joins nothing. The rule below exists to put a dot back on its
    // i, and a drawing has no dots -- applied to shapes it would swallow a
    // window into the wall above it purely for sitting under it.
    if (typeMode) {
      for (size_t i = 0; i < ids.size(); ++i) {
        for (size_t j = i + 1; j < ids.size(); ++j) {
          if (sameGlyph(comps[ids[i]], comps[ids[j]])) guf.unite(int(i), int(j));
        }
      }
    }
    std::vector<int> rootToGlyph(ids.size(), -1);
    auto& glyphs = lineGlyphs[li];
    for (size_t i = 0; i < ids.size(); ++i) {
      const int root = guf.find(int(i));
      if (rootToGlyph[root] < 0) {
        rootToGlyph[root] = int(glyphs.size());
        glyphs.push_back(Glyph{comps[ids[i]].bbox, comps[ids[i]].sx1, comps[ids[i]].sx2, {}});
      }
      Glyph& g = glyphs[rootToGlyph[root]];
      g.bbox.unionWith(comps[ids[i]].bbox);
      g.sx1 = std::min(g.sx1, comps[ids[i]].sx1);
      g.sx2 = std::max(g.sx2, comps[ids[i]].sx2);
      g.labels.push_back(ids[i]);
    }
    std::sort(glyphs.begin(), glyphs.end(),
              [](const Glyph& a, const Glyph& b) { return a.sx1 < b.sx1; });
  }

  // 8. Word breaks from gap statistics.
  //
  // The gap between two letters is their CLOSEST APPROACH, measured row by row:
  // for every row where both have ink, how far apart that row's ink is, taking
  // the smallest. This needs no slant at all, which is the point.
  //
  // Deskewing by one global angle assumes the whole title leans the same way. A
  // line that mixes an italic word with upright ones has no such angle -- the
  // estimate lands near zero, the italic words keep their lean, and their letters
  // read as far apart as words. Measured on tight caps with alternate words
  // sheared 16 degrees, that split 4 words into 8, and 12 by 22 degrees.
  //
  // Closest approach also describes a triangular glyph honestly. 'A' reaches back
  // under the letter before it at the baseline, so its axis-aligned box overlaps
  // its neighbour while the visible gap is wide; row by row, the letters are
  // compared where they actually come near each other.
  std::vector<std::vector<float>> lineGaps(lineGlyphs.size());
  std::vector<std::vector<float>> lineGapMin(lineGlyphs.size());
  {
    // Row extents per glyph, built from the label image one line at a time.
    std::vector<int> labelToGlyph(size_t(labelCount) + 1, -1);
    for (size_t li = 0; li < lineGlyphs.size(); ++li) {
      const auto& glyphs = lineGlyphs[li];
      if (glyphs.size() < 2) continue;

      std::fill(labelToGlyph.begin(), labelToGlyph.end(), -1);
      for (size_t gi = 0; gi < glyphs.size(); ++gi)
        for (int lab : glyphs[gi].labels)
          if (lab > 0 && size_t(lab) <= size_t(labelCount)) labelToGlyph[size_t(lab)] = int(gi);

      const int y0 = std::max(0, lineSpans[li].y1);
      const int y1 = std::min(h, lineSpans[li].y2);
      const int rows = std::max(0, y1 - y0);
      const size_t ng = glyphs.size();
      std::vector<int> lo(ng * size_t(rows), INT_MAX), hi(ng * size_t(rows), INT_MIN);

      for (int y = y0; y < y1; ++y) {
        const int32_t* row = &seg.labelImage[size_t(y) * w];
        for (int x = 0; x < w; ++x) {
          const int32_t lab = row[x];
          if (lab <= 0) continue;
          const int gi = labelToGlyph[size_t(lab)];
          if (gi < 0) continue;
          const size_t k = size_t(gi) * size_t(rows) + size_t(y - y0);
          lo[k] = std::min(lo[k], x);
          hi[k] = std::max(hi[k], x);
        }
      }

      lineGaps[li].assign(ng, 0.0f);
      lineGapMin[li].assign(ng, 0.0f);
      std::vector<int> perRow;
      for (size_t i = 1; i < ng; ++i) {
        perRow.clear();
        for (int r = 0; r < rows; ++r) {
          const size_t a = size_t(i - 1) * size_t(rows) + size_t(r);
          const size_t b = size_t(i) * size_t(rows) + size_t(r);
          if (hi[a] == INT_MIN || lo[b] == INT_MAX) continue;  // one has no ink here
          perRow.push_back(lo[b] - hi[a] - 1);
        }
        // A low percentile of the row gaps, not the strict minimum. The minimum
        // is where a triangular glyph's foot reaches back under its neighbour --
        // real, but not what separates words, and taking it merged 'Text' with
        // 'Animator'. The 10th percentile ignores that one extreme row while
        // still measuring where the letters genuinely come closest.
        int trueMin = INT_MAX;
        for (int g : perRow) trueMin = std::min(trueMin, g);
        lineGapMin[li][i] = trueMin == INT_MAX ? 0.0f : float(trueMin);

        int best = INT_MAX;
        if (!perRow.empty()) {
          const size_t k = size_t(0.10 * double(perRow.size() - 1) + 0.5);
          std::nth_element(perRow.begin(), perRow.begin() + k, perRow.end());
          best = perRow[k];
        }
        // No shared row at all -- a superscript, say. Fall back to the deskewed
        // extents, which is the best available answer when the two never meet.
        lineGaps[li][i] = best == INT_MAX ? std::max(0.0f, glyphs[i].sx1 - glyphs[i - 1].sx2)
                                          : float(best);
      }
    }
  }

  // Gaps are compared as a FRACTION OF LOCAL LETTER SIZE, never in raw pixels.
  //
  // A pixel means nothing on its own: 20px is a tight letter gap in a 200px face
  // and a wide word space in a 60px one. What is stable across typefaces is the
  // ratio -- a word space runs about a quarter to a third of the letter height in
  // almost every face, while letter spacing is far below that.
  //
  // Normalising per adjacency is what makes mixed fonts work ANYWHERE, including
  // two faces inside one line. A per-line threshold would not: a line can change
  // font partway, and then the line has no single answer either.
  //
  // The scale comes from the glyphs immediately around the gap, so it tracks a
  // font change as it happens rather than averaging across one.
  std::vector<std::vector<float>> lineNorm(lineGlyphs.size());
  std::vector<int> allNorm;  // normalised gaps, x1000, for clustering
  for (size_t li = 0; li < lineGlyphs.size(); ++li) {
    const auto& glyphs = lineGlyphs[li];
    lineNorm[li].assign(glyphs.size(), 0.0f);
    for (size_t i = 1; i < glyphs.size(); ++i) {
      // Median height of the glyphs around this gap. Median, not mean, so one
      // tall cap or a stray comma does not drag the scale.
      std::vector<int> hs;
      const size_t lo = i >= 3 ? i - 3 : 0;
      const size_t hi = std::min(glyphs.size(), i + 3);
      for (size_t k = lo; k < hi; ++k) hs.push_back(std::max(1, glyphs[k].bbox.height()));
      std::nth_element(hs.begin(), hs.begin() + hs.size() / 2, hs.end());
      const float scale = float(std::max(1, hs[hs.size() / 2]));

      const float n = std::max(0.0f, lineGaps[li][i]) / scale;
      lineNorm[li][i] = n;
      // The closest approach in the same units, for the manual threshold.
      lineGapMin[li][i] /= scale;
      allNorm.push_back(int(n * 1000.0f));
    }
  }

  // 8b. Reference text reconciliation.
  //
  // OPT-IN, and inert when the string is empty: everything below this block is
  // reached unchanged in that case. This is a layer over a detector that already
  // works, not a replacement for it.
  //
  // Two things happen, in this order:
  //
  //   MERGE. A character can render as several disconnected marks that the glyph
  //   rule cannot rejoin -- a '"' is two marks SIDE BY SIDE, so they share a
  //   vertical band and the diacritic rule (x-overlap plus vertical separation,
  //   which handles 'i' and umlauts) never fires. The marks of each line are
  //   aligned against the CHARACTERS of the string, so the merge lands where the
  //   text says a multi-stroke character is -- not merely where the pixels happen
  //   to be closest. Closest-first is not good enough: an 'r' kerns tighter
  //   against a following quote than the quote's own strokes are spaced, which
  //   fuses the 'r' to half the quote and strands the other half.
  //
  //   CUT. Word boundaries then come from the string's spaces. No gap statistic
  //   is consulted -- not Otsu, not Word Gap, not Character Gap.
  //
  // Merging fixes OVER-splitting only. Two characters that touched and labelled
  // as one mark cannot be recovered this way, and guessing a split position
  // inside a mark would be inventing data, so that case reports and falls back.
  std::vector<std::vector<int>> refWordStarts;  // per line, first glyph of each word
  bool refApplied = false;
  if (!params.referenceText.empty() && typeMode) {
    // Each word is a list of per-character mark allowances, so a word's size is
    // its character count and its contents drive the alignment below.
    const std::vector<std::vector<CharSpec>> words = referenceWords(params.referenceText);

    // Lines that actually carry marks. A line span always has ink when it is
    // found, but a line whose every component was despeckled away has none, and
    // it must not claim a word.
    std::vector<size_t> rows;
    for (size_t li = 0; li < lineGlyphs.size(); ++li)
      if (!lineGlyphs[li].empty()) rows.push_back(li);

    const size_t K = rows.size(), N = words.size();
    std::vector<int> marks(K, 0);
    long long totalMarks = 0, totalChars = 0;
    for (size_t i = 0; i < K; ++i) {
      marks[i] = int(lineGlyphs[rows[i]].size());
      totalMarks += marks[i];
    }
    for (const auto& wd : words) totalChars += int(wd.size());

    // Prefix sums, so a chunk's character count is one subtraction.
    std::vector<long long> pre(N + 1, 0);
    for (size_t j = 0; j < N; ++j) pre[j + 1] = pre[j] + int(words[j].size());

    std::string fail;
    if (N == 0) {
      fail = "the reference text has no words";
    } else if (N < K) {
      fail = "the reference text has " + std::to_string(N) + " word" + (N == 1 ? "" : "s") +
             " for " + std::to_string(K) + " detected lines";
    } else if (totalChars > totalMarks) {
      // Say WHY, because the count alone leaves nothing to act on. Marks falling
      // short of characters always means letters ran together: an 'ff' ligature,
      // a script face, or tracking tight enough to close the gap.
      fail = "the text has " + std::to_string(totalChars) + " characters but " +
             std::to_string(totalMarks) +
             " marks were found - joined letters (a ligature, or a script face) count as one";
    }

    // Words pack onto lines in reading order: a wrapped line breaks at a word
    // boundary, so each line takes a contiguous run of words. Marks can only ever
    // EXCEED characters, never fall short, so a line's run must fit within its
    // mark count -- and every line must take at least one word, since a detected
    // line has ink in it.
    //
    // feasible[i][j] asks whether lines i.. can consume words j.. . Built
    // backwards, it turns the search into a walk that can never enter a dead end.
    std::vector<std::vector<char>> feasible;
    if (fail.empty()) {
      feasible.assign(K + 1, std::vector<char>(N + 1, 0));
      feasible[K][N] = 1;
      for (size_t i = K; i-- > 0;) {
        for (size_t j = 0; j < N; ++j) {
          for (size_t e = j + 1; e <= N; ++e) {
            if (pre[e] - pre[j] > marks[i]) break;  // only grows from here
            if (feasible[i + 1][e]) {
              feasible[i][j] = 1;
              break;
            }
          }
        }
      }
      if (!feasible[0][0]) {
        // Name where it dies rather than restating the totals. Forward
        // reachability ignoring feasibility: the deepest line any packing can
        // reach at all is the one that could not be satisfied.
        std::vector<char> reach(N + 1, 0);
        reach[0] = 1;
        size_t at = 0;
        for (; at < K; ++at) {
          std::vector<char> next(N + 1, 0);
          bool advanced = false;
          for (size_t j = 0; j < N; ++j) {
            if (!reach[j]) continue;
            for (size_t e = j + 1; e <= N && pre[e] - pre[j] <= marks[at]; ++e) {
              next[e] = 1;
              advanced = true;
            }
          }
          if (!advanced) break;
          reach.swap(next);
        }
        const size_t line = std::min(at, K - 1);
        fail = "line " + std::to_string(line + 1) + " has " + std::to_string(marks[line]) +
               " marks, too few for the words the text puts on it";
      }
    }

    // The characters a packing puts on one line, flattened out of its words.
    auto lineChars = [&](size_t from, size_t to) {
      std::vector<CharSpec> spec;
      for (size_t wi = from; wi < to; ++wi)
        spec.insert(spec.end(), words[wi].begin(), words[wi].end());
      return spec;
    };

    // Aligning one line's MARKS against its CHARACTERS, left to right.
    //
    // Every character takes at least one mark, and takes several only where the
    // string says it may -- a quote is allowed two strokes, an ordinary letter is
    // not. That is the whole point of being handed the text: the merge lands on
    // the character that is genuinely made of two pieces, instead of wherever the
    // pixels happen to sit closest. An 'r' followed by a closing quote is exactly
    // where closest-first goes wrong, because the kern is tighter than the gap
    // inside the quote itself.
    //
    // A character takes several marks only when the count AND the arrangement
    // both fit -- a quote's two strokes stand side by side at matching heights, a
    // dot sits clear above its stem. Without the arrangement test the allowance
    // gets spent on whatever happens to be adjacent, which is how a trailing full
    // stop was able to move the merges to the wrong place entirely.
    //
    // Unexpected merges are not forbidden, only made expensive: a face that
    // shatters a letter no different from any other still reconciles, it simply
    // loses to any reading that does not have to. Among readings of equal cost
    // the narrower gaps win, which is the old behaviour and the right tie-break.
    //
    // Returns the first mark of each character, and the cost of the reading;
    // `starts` is left empty when the line cannot be read at all.
    struct Reading {
      std::vector<int> starts;
      double cost = 0.0;
    };
    auto alignLine = [&](size_t li, const std::vector<CharSpec>& spec) {
      const size_t M = lineGlyphs[li].size(), C = spec.size();
      Reading out;
      if (C == 0 || C > M) return out;

      // Cost of a character taking a mark it was never entitled to, or taking one
      // that does not sit the way its parts do. Normalised gaps live below ~2, so
      // this cannot be outweighed by any sum of them.
      const double kUnexpected = 1000.0;
      const size_t slack = M - C;  // no character can take more than this extra

      const double kInf = std::numeric_limits<double>::infinity();
      std::vector<double> cost((C + 1) * (M + 1), kInf);
      std::vector<int> take((C + 1) * (M + 1), 0);
      auto at = [&](size_t j, size_t s) { return j * (M + 1) + s; };
      cost[at(0, 0)] = 0.0;

      for (size_t j = 0; j < C; ++j) {
        for (size_t s = 0; s <= M; ++s) {
          const double base = cost[at(j, s)];
          if (base == kInf) continue;
          const size_t maxK = std::min(slack + 1, M - s);
          double gapSum = 0.0;
          bool fits = true;  // do the marks so far sit the way this character's do
          for (size_t k = 1; k <= maxK; ++k) {
            // Marks [s, s+k) form this character; its internal gaps are the ones
            // it swallows, and each newly added mark must sit against the one
            // before it the way this character's parts do.
            if (k > 1) {
              gapSum += double(lineNorm[li][s + k - 1]);
              fits = fits && shapeFits(spec[j].shape, lineGlyphs[li][s + k - 2].bbox,
                                       lineGlyphs[li][s + k - 1].bbox);
            }
            const double extra = (int(k) > spec[j].allow || !fits)
                                     ? kUnexpected * double(int(k) - 1)
                                     : 0.0;
            const double c = base + gapSum + extra;
            if (c < cost[at(j + 1, s + k)]) {
              cost[at(j + 1, s + k)] = c;
              take[at(j + 1, s + k)] = int(k);
            }
          }
        }
      }
      if (cost[at(C, M)] == kInf) return out;

      out.cost = cost[at(C, M)];
      out.starts.assign(C, 0);
      size_t s = M;
      for (size_t j = C; j-- > 0;) {
        s -= size_t(take[at(j + 1, s)]);
        out.starts[j] = int(s);
      }
      return out;
    };

    // A candidate packing, and how well the pixels agree with it.
    struct Packing {
      std::vector<size_t> bounds;  // word index each line starts at, plus N
      double cost = 0.0;
    };

    std::vector<Packing> best;
    if (fail.empty()) {
      // When nothing over-split, every line must be filled to the character, so
      // the packing is forced and the search below finds the one answer. Slack
      // admits a handful more, and the alignment cost breaks the tie -- the
      // packing whose merges the string actually accounts for.
      const int kMaxCandidates = 256;
      int seen = 0;

      auto score = [&](const std::vector<size_t>& bounds) {
        double total = 0.0;
        for (size_t i = 0; i < K; ++i) {
          const Reading r = alignLine(rows[i], lineChars(bounds[i], bounds[i + 1]));
          if (r.starts.empty()) return std::numeric_limits<double>::infinity();
          total += r.cost;
        }
        return total;
      };

      std::vector<size_t> bounds(K + 1, 0);
      bounds[K] = N;
      std::function<void(size_t, size_t)> walk = [&](size_t i, size_t j) {
        if (seen >= kMaxCandidates) return;
        if (i == K) {
          if (j != N) return;
          ++seen;
          Packing p;
          p.bounds = bounds;
          p.cost = score(bounds);
          best.push_back(std::move(p));
          return;
        }
        for (size_t e = j + 1; e <= N && pre[e] - pre[j] <= marks[i]; ++e) {
          if (!feasible[i + 1][e]) continue;
          bounds[i + 1] = e;
          walk(i + 1, e);
          if (seen >= kMaxCandidates) return;
        }
      };
      bounds[0] = 0;
      walk(0, 0);
      if (best.empty()) fail = "the reference text could not be matched to the detected lines";
    }

    // Cheapest reading wins; ties go to the packing found first, which is the
    // one that fills the earlier lines fullest -- what wrapping does.
    size_t pick = 0;
    for (size_t i = 1; i < best.size(); ++i)
      if (best[i].cost < best[pick].cost) pick = i;

    // Does the reading actually DESCRIBE THIS PICTURE?
    //
    // Everything above only asks whether the words FIT: enough of them for the
    // lines, no more characters than marks, each line's run short enough. That is
    // arithmetic, and arithmetic is happy with any text of roughly the right
    // length -- so a completely wrong string was accepted in silence and the
    // words were cut wherever it said. Fitting is not matching.
    //
    // What a wrong string cannot fake is WHERE ITS SPACES LAND. A space in the
    // string claims a gap in the picture, and gaps are not interchangeable: the
    // ones between words are plainly wider than the ones between letters, or a
    // reader could not tell the words apart either. Text that does not belong to
    // this picture scatters its claims over ordinary letter gaps.
    if (fail.empty()) {
      const std::vector<size_t>& b0 = best[pick].bounds;
      for (size_t i = 0; i < K && fail.empty(); ++i) {
        const size_t li = rows[i];
        const Reading r = alignLine(li, lineChars(b0[i], b0[i + 1]));
        if (r.starts.empty()) continue;

        // Which character indices start a word, per the string.
        std::vector<char> startsWord(r.starts.size(), 0);
        {
          size_t c = 0;
          for (size_t wi = b0[i]; wi < b0[i + 1]; ++wi) {
            if (c < startsWord.size()) startsWord[c] = 1;
            c += words[wi].size();
          }
        }

        std::vector<float> cutGaps, innerGaps;
        for (size_t j = 1; j < r.starts.size(); ++j) {
          const size_t bnd = size_t(r.starts[j]);
          if (bnd == 0 || bnd >= lineGlyphs[li].size()) continue;
          if (startsWord[j]) {
            // A word break needs a blank column to sit in. If the two marks
            // overlap once the italic lean is taken out, there is none -- no
            // space was ever set there, whatever the string says. Geometry, not
            // statistics, so this one is decisive.
            if (lineGlyphs[li][bnd].sx1 - lineGlyphs[li][bnd - 1].sx2 <= 0.0f) {
              fail = "line " + std::to_string(i + 1) +
                     ": the text puts a word break where two letters touch";
              break;
            }
            cutGaps.push_back(lineNorm[li][bnd]);
          } else {
            innerGaps.push_back(lineNorm[li][bnd]);
          }
        }
        if (!fail.empty()) break;

        // Compared as MEDIANS, and only with enough of both to mean anything. A
        // single odd gap -- after a full stop, or across a font change -- should
        // not condemn a string that is otherwise plainly right.
        if (cutGaps.size() >= 2 && innerGaps.size() >= 3) {
          auto median = [](std::vector<float>& v) {
            std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
            return v[v.size() / 2];
          };
          const float cut = median(cutGaps), inner = median(innerGaps);
          // Generous on purpose. Correcting a face whose word spaces are unusually
          // tight is one of the reasons to type the text in at all, so the bar is
          // "wider than a letter gap", not "as wide as a typical word space".
          if (cut < 1.5f * inner) {
            fail = "the word breaks in the text do not line up with the spaces in "
                   "the picture";
          }
        }
      }
    }

    if (fail.empty()) {
      const std::vector<size_t>& bounds = best[pick].bounds;

      // Commit. Only now is anything mutated, so a failure above leaves the
      // automatic result exactly as it was.
      refWordStarts.assign(lineGlyphs.size(), {});
      for (size_t i = 0; i < K; ++i) {
        const size_t li = rows[i];
        const Reading r = alignLine(li, lineChars(bounds[i], bounds[i + 1]));

        // Fuse the run of marks each character claimed -- the same union §7 does
        // when it joins a dot to its stem.
        const std::vector<int>& starts = r.starts;
        if (starts.empty()) continue;  // scoring already proved this reads

        std::vector<Glyph> merged;
        std::vector<float> nrm, gmin, gaps;
        merged.reserve(starts.size());
        for (size_t k = 0; k < starts.size(); ++k) {
          const size_t from = size_t(starts[k]);
          const size_t to = k + 1 < starts.size() ? size_t(starts[k + 1]) : lineGlyphs[li].size();
          Glyph g = lineGlyphs[li][from];
          for (size_t m = from + 1; m < to; ++m) {
            g.bbox.unionWith(lineGlyphs[li][m].bbox);
            g.sx1 = std::min(g.sx1, lineGlyphs[li][m].sx1);
            g.sx2 = std::max(g.sx2, lineGlyphs[li][m].sx2);
            g.labels.insert(g.labels.end(), lineGlyphs[li][m].labels.begin(),
                            lineGlyphs[li][m].labels.end());
          }
          merged.push_back(std::move(g));
          // A surviving boundary keeps the gap it always had; merging never
          // changes the distance between what is left.
          const size_t b = k == 0 ? 0 : size_t(starts[k]);
          nrm.push_back(k == 0 ? 0.0f : lineNorm[li][b]);
          gmin.push_back(k == 0 ? 0.0f : lineGapMin[li][b]);
          gaps.push_back(k == 0 ? 0.0f : lineGaps[li][b]);
        }
        lineGlyphs[li].swap(merged);
        lineNorm[li].swap(nrm);
        lineGapMin[li].swap(gmin);
        lineGaps[li].swap(gaps);

        // Word cuts, in the merged numbering: the string's spaces, nothing else.
        std::vector<int>& ws = refWordStarts[li];
        int at = 0;
        for (size_t wi = bounds[i]; wi < bounds[i + 1]; ++wi) {
          ws.push_back(at);
          at += int(words[wi].size());
        }
      }
      refApplied = true;
      seg.refStatus = RefStatus::Applied;
    } else {
      // Loud, and the automatic detection stands. A silent fallback would look
      // like the string had been honoured.
      seg.refStatus = RefStatus::Failed;
      seg.refMessage = fail;
    }
  }

  // A typeface sets two kinds of gap -- between letters, and between words --
  // and in normalised units the two clusters land in the same place whatever the
  // face. Otsu's method finds the split between them directly.
  auto otsuOf = [](std::vector<int> gaps) -> float {
    if (gaps.size() < 6) return 0.0f;
    int maxGap = 0;
    for (int g : gaps) maxGap = std::max(maxGap, g);
    if (maxGap <= 0) return 0.0f;

    std::vector<int> hist(size_t(maxGap) + 1, 0);
    for (int g : gaps) ++hist[size_t(g)];

    const double total = double(gaps.size());
    double sumAll = 0.0;
    for (int v = 0; v <= maxGap; ++v) sumAll += double(v) * hist[size_t(v)];

    double wB = 0.0, sumB = 0.0, bestVar = -1.0, bestM0 = 0.0, bestM1 = 0.0;
    for (int t = 0; t < maxGap; ++t) {
      wB += hist[size_t(t)];
      if (wB <= 0.0) continue;
      const double wF = total - wB;
      if (wF <= 0.0) break;
      sumB += double(t) * hist[size_t(t)];
      const double m0 = sumB / wB;             // letter gaps
      const double m1 = (sumAll - sumB) / wF;  // word gaps
      const double var = wB * wF * (m0 - m1) * (m0 - m1);
      if (var > bestVar) {
        bestVar = var;
        bestM0 = m0;
        bestM1 = m1;
      }
    }

    // Midway between the cluster means, not the Otsu index: the index sits at the
    // top edge of the letter-gap cluster and its tail spills over it, breaking
    // ordinary letters into words.
    //
    // Only trusted when the clusters are genuinely apart. A single-word line has
    // letter gaps only, and Otsu will split those down the middle and shatter it.
    if (bestM1 < 2.0 * std::max(bestM0, 1.0)) return 0.0f;
    return float(0.5 * (bestM0 + bestM1)) / 1000.0f;
  };

  // Fraction of letter height that separates words when the gaps themselves are
  // too few or too uniform to cluster -- a single word, or one short line.
  const float kFallbackNorm = 0.28f;
  const float otsuNorm = otsuOf(allNorm);
  const float breakNorm = otsuNorm > 0.0f ? otsuNorm : kFallbackNorm;

  // 9. Emit groups.
  auto pushGroup = [&](const RectI& bbox, float sx1, float sx2, int glyphs, int line, int idx,
                       int firstGlyph, std::vector<int> labels) {
    Group g;
    g.bbox = bbox;
    g.sx1 = sx1;
    g.sx2 = sx2;
    g.glyphCount = std::max(1, glyphs);
    g.firstGlyph = firstGlyph;
    g.line = line;
    g.indexInLine = idx;
    g.labels = std::move(labels);
    seg.groups.push_back(std::move(g));
  };

  for (size_t li = 0; li < lineGlyphs.size(); ++li) {
    const auto& glyphs = lineGlyphs[li];
    if (glyphs.empty()) continue;
    // Normalised units: a fraction of local letter height, not pixels.
    const float breakAt = params.wordGapSensitivity * breakNorm;

    switch (params.mode) {
      // One connected region, one element. Identical emission to Character mode;
      // what makes it Object mode happened above, where nothing was joined.
      case GroupMode::Object:
      case GroupMode::Character:
        for (size_t i = 0; i < glyphs.size(); ++i)
          pushGroup(glyphs[i].bbox, glyphs[i].sx1, glyphs[i].sx2, 1, int(li), int(i), int(i),
                    glyphs[i].labels);
        break;

      case GroupMode::Line: {
        RectI box;
        float lo = glyphs[0].sx1, hi = glyphs[0].sx2;
        std::vector<int> labels;
        for (const auto& g : glyphs) {
          box.unionWith(g.bbox);
          lo = std::min(lo, g.sx1);
          hi = std::max(hi, g.sx2);
          labels.insert(labels.end(), g.labels.begin(), g.labels.end());
        }
        pushGroup(box, lo, hi, int(glyphs.size()), int(li), 0, 0, std::move(labels));
        break;
      }

      case GroupMode::Word: {
        // With reference text the cuts are already known -- they are the spaces
        // in the string -- so none of the gap machinery below is consulted.
        if (refApplied) {
          const std::vector<int>& ws = refWordStarts[li];
          for (size_t k = 0; k < ws.size(); ++k) {
            const size_t from = size_t(ws[k]);
            const size_t to = k + 1 < ws.size() ? size_t(ws[k + 1]) : glyphs.size();
            if (from >= glyphs.size() || to <= from) continue;
            RectI box = glyphs[from].bbox;
            float lo = glyphs[from].sx1, hi = glyphs[from].sx2;
            std::vector<int> labels = glyphs[from].labels;
            for (size_t j = from + 1; j < to && j < glyphs.size(); ++j) {
              box.unionWith(glyphs[j].bbox);
              lo = std::min(lo, glyphs[j].sx1);
              hi = std::max(hi, glyphs[j].sx2);
              labels.insert(labels.end(), glyphs[j].labels.begin(), glyphs[j].labels.end());
            }
            pushGroup(box, lo, hi, int(std::min(to, glyphs.size()) - from), int(li), int(k),
                      int(from), std::move(labels));
          }
          break;
        }

        RectI box = glyphs[0].bbox;
        float lo = glyphs[0].sx1, hi = glyphs[0].sx2;
        std::vector<int> labels = glyphs[0].labels;
        int idx = 0, held = 1;
        size_t first = 0;
        for (size_t i = 1; i < glyphs.size(); ++i) {
          // Max Letter Gap, when set, decides this outright: a gap wider than it
          // starts a new word. Same grouping bridging would produce, but the
          // letters are never fused, so character boundaries -- and the manual
          // splits that name them -- survive.
          bool brk = params.maxLetterGap > 0.0f ? lineGapMin[li][i] > params.maxLetterGap
                                                : lineNorm[li][i] > breakAt;

          // Overlapping letters are never a word break, whatever the statistics
          // say. If the two extents overlap there is no blank column between
          // them, so no space was ever set there.
          //
          // This is not redundant with the row-wise gap. That gap is the tenth
          // percentile of the rows, not the minimum -- the minimum is where a
          // triangular glyph's foot reaches under its neighbour, and taking it
          // merged real words. But in a thin face an 'A' meets the 'T' after it
          // only at the very top and very bottom, and those are exactly the rows
          // the percentile discards, so ANIMATION split into ANIMA and TION with
          // the two boxes visibly overlapping on screen.
          //
          // Measured deskewed, because in an italic every letter overlaps its
          // neighbour in raw x and the test would never fire.
          if (glyphs[i].sx1 - glyphs[i - 1].sx2 <= 0.0f) brk = false;
          if (brk) {
            pushGroup(box, lo, hi, held, int(li), idx++, int(first), std::move(labels));
            first = i;
            box = glyphs[i].bbox;
            lo = glyphs[i].sx1;
            hi = glyphs[i].sx2;
            labels = glyphs[i].labels;
            held = 1;
          } else {
            ++held;
            box.unionWith(glyphs[i].bbox);
            lo = std::min(lo, glyphs[i].sx1);
            hi = std::max(hi, glyphs[i].sx2);
            labels.insert(labels.end(), glyphs[i].labels.begin(), glyphs[i].labels.end());
          }
        }
        pushGroup(box, lo, hi, held, int(li), idx, int(first), std::move(labels));
        break;
      }
    }
  }

  // A global character index for every glyph, in reveal-reading order. This is
  // the currency both overrides speak and the number Show Detection paints.
  std::vector<std::vector<int>> glyphIndexOf(lineGlyphs.size());
  {
    int next = 0;
    for (size_t li = 0; li < lineGlyphs.size(); ++li) {
      glyphIndexOf[li].resize(lineGlyphs[li].size());
      for (size_t j = 0; j < lineGlyphs[li].size(); ++j) glyphIndexOf[li][j] = next++;
    }
  }

  // 10. Manual overrides.
  //
  // Grouping a line is a choice of BOUNDARIES between its characters, so both
  // overrides are edits to that set -- a merge removes one, a split adds one.
  // Expressing them the same way is what lets them combine freely, in any order,
  // and makes each independent of what the automatic pass decided.
  //
  // Addressed by character, never by unit. Unit numbers shift the moment a merge
  // is applied, so a second correction to the same line becomes inexpressible;
  // character numbers never move.
  //
  // This exists because no automatic rule survives every title: a face whose
  // letter spacing is as wide as another's word spacing cannot be separated by
  // any threshold that also suits the rest of the frame.
  if (!params.mergeAt.empty() || !params.splitBefore.empty()) {
    // An index in both lists cancels. Asking for a boundary to be removed AND
    // added at the same character says nothing, so the automatic decision stands
    // -- which is what the pair of edits amounts to anyway.
    //
    // Without this the order of application decides, so the two lists would
    // quietly disagree and the same pair of numbers would mean different things
    // depending on which pass ran last.
    std::vector<int> merges, splits;
    for (int ci : params.mergeAt)
      if (std::find(params.splitBefore.begin(), params.splitBefore.end(), ci) ==
          params.splitBefore.end())
        merges.push_back(ci);
    for (int ci : params.splitBefore)
      if (std::find(params.mergeAt.begin(), params.mergeAt.end(), ci) == params.mergeAt.end())
        splits.push_back(ci);

    const size_t nLines = lineGlyphs.size();
    std::vector<std::vector<int>> starts(nLines);
    for (const Group& g : seg.groups)
      if (g.line >= 0 && size_t(g.line) < nLines) starts[size_t(g.line)].push_back(g.firstGlyph);

    // Map a character index back to its line and position.
    auto locate = [&](int ci, size_t* li, size_t* gi) {
      for (size_t l = 0; l < nLines; ++l)
        for (size_t j = 0; j < glyphIndexOf[l].size(); ++j)
          if (glyphIndexOf[l][j] == ci) {
            *li = l;
            *gi = j;
            return true;
          }
      return false;
    };

    for (int ci : merges) {
      size_t li = 0, gi = 0;
      if (!locate(ci, &li, &gi) || gi == 0) continue;  // a line always starts a unit
      auto& v = starts[li];
      v.erase(std::remove(v.begin(), v.end(), int(gi)), v.end());
    }
    for (int ci : splits) {
      size_t li = 0, gi = 0;
      if (!locate(ci, &li, &gi) || gi == 0) continue;
      starts[li].push_back(int(gi));
    }

    std::vector<Group> rebuilt;
    for (size_t li = 0; li < nLines; ++li) {
      const auto& glyphs = lineGlyphs[li];
      if (glyphs.empty()) continue;
      auto& v = starts[li];
      v.push_back(0);  // a line always begins a unit
      std::sort(v.begin(), v.end());
      v.erase(std::unique(v.begin(), v.end()), v.end());

      for (size_t k = 0; k < v.size(); ++k) {
        const size_t from = size_t(std::max(0, v[k]));
        const size_t to = k + 1 < v.size() ? size_t(v[k + 1]) : glyphs.size();
        if (from >= glyphs.size() || to <= from) continue;

        Group g;
        g.bbox = glyphs[from].bbox;
        g.sx1 = glyphs[from].sx1;
        g.sx2 = glyphs[from].sx2;
        g.labels = glyphs[from].labels;
        for (size_t j = from + 1; j < to && j < glyphs.size(); ++j) {
          g.bbox.unionWith(glyphs[j].bbox);
          g.sx1 = std::min(g.sx1, glyphs[j].sx1);
          g.sx2 = std::max(g.sx2, glyphs[j].sx2);
          g.labels.insert(g.labels.end(), glyphs[j].labels.begin(), glyphs[j].labels.end());
        }
        g.glyphCount = int(std::min(to, glyphs.size()) - from);
        g.firstGlyph = int(from);
        g.line = int(li);
        rebuilt.push_back(std::move(g));
      }
    }

    int line = -1, idx = 0;
    for (auto& g : rebuilt) {
      if (g.line != line) {
        line = g.line;
        idx = 0;
      }
      g.indexInLine = idx++;
    }
    if (!rebuilt.empty()) seg.groups.swap(rebuilt);
  }

  // Character indices for every unit, after any override, so the label and the
  // overrides always agree about what a number means.
  for (Group& g : seg.groups) {
    g.glyphIndices.clear();
    g.glyphStarts.clear();
    if (g.line < 0 || size_t(g.line) >= lineGlyphs.size()) continue;
    const auto& gl = lineGlyphs[size_t(g.line)];
    for (int j = 0; j < g.glyphCount; ++j) {
      const size_t k = size_t(g.firstGlyph) + size_t(j);
      if (k >= gl.size()) continue;
      g.glyphStarts.push_back(gl[k].bbox.x1);
      g.glyphIndices.push_back(glyphIndexOf[size_t(g.line)][k]);
    }
    g.glyphIndex = g.glyphIndices.empty() ? 0 : g.glyphIndices.front();
  }

  seg.labelToGroup.assign(size_t(labelCount) + 1, -1);
  for (size_t gi = 0; gi < seg.groups.size(); ++gi)
    for (int l : seg.groups[gi].labels) seg.labelToGroup[l] = int(gi);

  return seg;
}

}  // namespace rta
