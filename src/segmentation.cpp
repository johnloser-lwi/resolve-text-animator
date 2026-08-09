#include "segmentation.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <numeric>

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
  std::vector<int> px, py;
  px.reserve(size_t(w) * h / 16);
  for (int y = 0; y < h; y += 3) {
    const uint8_t* row = &mask[size_t(y) * w];
    for (int x = 0; x < w; ++x) {
      if (!row[x]) continue;
      px.push_back(x);
      py.push_back(y);
    }
  }
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

  // 2. Optional bridge dilation. Labels are computed on the dilated mask but
  //    written back only onto true text pixels, so bboxes stay accurate.
  std::vector<uint8_t> dilated;
  if (params.bridgeRadius > 0) dilated = dilate(mask, w, h, params.bridgeRadius);
  const std::vector<uint8_t>& labelMask = params.bridgeRadius > 0 ? dilated : mask;

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
  const float slantTan = params.autoSlant ? detectSlant(mask, w, h)
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
  for (int l = 1; l <= labelCount; ++l) comps[l].alive = comps[l].area >= params.minBlobArea;
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
    for (size_t i = 0; i < ids.size(); ++i) {
      for (size_t j = i + 1; j < ids.size(); ++j) {
        if (sameGlyph(comps[ids[i]], comps[ids[j]])) guf.unite(int(i), int(j));
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
  std::vector<std::vector<int>> lineGapMin(lineGlyphs.size());
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
      lineGapMin[li].assign(ng, 0);
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
        lineGapMin[li][i] = trueMin == INT_MAX ? 0 : trueMin;

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
      allNorm.push_back(int(n * 1000.0f));
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
          bool brk = params.maxLetterGap > 0 ? lineGapMin[li][i] > params.maxLetterGap
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

    for (int ci : params.mergeAt) {
      size_t li = 0, gi = 0;
      if (!locate(ci, &li, &gi) || gi == 0) continue;  // a line always starts a unit
      auto& v = starts[li];
      v.erase(std::remove(v.begin(), v.end(), int(gi)), v.end());
    }
    for (int ci : params.splitBefore) {
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
