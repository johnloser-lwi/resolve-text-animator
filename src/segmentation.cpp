#include "segmentation.h"

#include <algorithm>
#include <cmath>
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

// Separable max filter. Used to bridge the hairline joins of script fonts so
// that a cursive word labels as one component instead of shattering.
std::vector<uint8_t> dilate(const std::vector<uint8_t>& src, int w, int h, int r) {
  std::vector<uint8_t> tmp(size_t(w) * h, 0);
  std::vector<uint8_t> out(size_t(w) * h, 0);
  for (int y = 0; y < h; ++y) {
    const uint8_t* s = &src[size_t(y) * w];
    uint8_t* d = &tmp[size_t(y) * w];
    for (int x = 0; x < w; ++x) {
      uint8_t m = 0;
      const int lo = std::max(0, x - r), hi = std::min(w - 1, x + r);
      for (int i = lo; i <= hi && !m; ++i) m = s[i];
      d[x] = m;
    }
  }
  for (int x = 0; x < w; ++x) {
    for (int y = 0; y < h; ++y) {
      uint8_t m = 0;
      const int lo = std::max(0, y - r), hi = std::min(h - 1, y + r);
      for (int i = lo; i <= hi && !m; ++i) m = tmp[size_t(i) * w + x];
      out[size_t(y) * w + x] = m;
    }
  }
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
  std::vector<int> allGaps;
  for (const auto& glyphs : lineGlyphs) {
    for (size_t i = 1; i < glyphs.size(); ++i)
      allGaps.push_back(int(std::max(0.0f, glyphs[i].sx1 - glyphs[i - 1].sx2)));
  }
  float medianGap = 0.0f;
  if (!allGaps.empty()) {
    std::nth_element(allGaps.begin(), allGaps.begin() + allGaps.size() / 2, allGaps.end());
    medianGap = float(allGaps[allGaps.size() / 2]);
  }

  // A typeface sets two kinds of gap -- between letters, and between words --
  // and they form two clusters. Otsu's method finds the split between them
  // directly, which is what makes the same setting work across faces.
  //
  // A fixed multiple of the line height cannot: a condensed bold caps face has
  // tall glyphs and tight tracking, so a height-derived floor lands ABOVE its
  // word gaps and the whole line merges. Measured on one such title, no single
  // sensitivity worked for it and for a lighter, wider face at the same time.
  float otsuGap = 0.0f;
  if (allGaps.size() >= 6) {
    int maxGap = 0;
    for (int g : allGaps) maxGap = std::max(maxGap, g);
    if (maxGap > 0) {
      std::vector<int> hist(size_t(maxGap) + 1, 0);
      for (int g : allGaps) ++hist[size_t(g)];

      const double total = double(allGaps.size());
      double sumAll = 0.0;
      for (int v = 0; v <= maxGap; ++v) sumAll += double(v) * hist[size_t(v)];

      double wB = 0.0, sumB = 0.0, bestVar = -1.0;
      int bestT = 0;
      double bestM0 = 0.0, bestM1 = 0.0;
      for (int t = 0; t < maxGap; ++t) {
        wB += hist[size_t(t)];
        if (wB <= 0.0) continue;
        const double wF = total - wB;
        if (wF <= 0.0) break;
        sumB += double(t) * hist[size_t(t)];
        const double m0 = sumB / wB;              // letter gaps
        const double m1 = (sumAll - sumB) / wF;   // word gaps
        const double var = wB * wF * (m0 - m1) * (m0 - m1);
        if (var > bestVar) {
          bestVar = var;
          bestT = t;
          bestM0 = m0;
          bestM1 = m1;
        }
      }

      // Only trust it when the two clusters are genuinely apart. A single-word
      // line has letter gaps only, and Otsu will happily split those down the
      // middle and shatter the word.
      // Midway between the two cluster means, not the Otsu index itself. The
      // index sits at the top edge of the letter-gap cluster, so the tail of
      // that cluster spills over it and ordinary letters break as words.
      (void)bestT;
      if (bestM1 >= 2.0 * std::max(bestM0, 1.0)) otsuGap = float(0.5 * (bestM0 + bestM1));
    }
  }

  // 9. Emit groups.
  auto pushGroup = [&](const RectI& bbox, float sx1, float sx2, int line, int idx,
                       std::vector<int> labels) {
    Group g;
    g.bbox = bbox;
    g.sx1 = sx1;
    g.sx2 = sx2;
    g.line = line;
    g.indexInLine = idx;
    g.labels = std::move(labels);
    seg.groups.push_back(std::move(g));
  };

  for (size_t li = 0; li < lineGlyphs.size(); ++li) {
    const auto& glyphs = lineGlyphs[li];
    if (glyphs.empty()) continue;
    const float lineHeight = float(lineSpans[li].height());
    // Prefer the split the gaps themselves show. The height-derived floor stays
    // as the fallback for when they show nothing usable -- a single word, or a
    // face whose letter and word spacing genuinely overlap.
    const float breakAt = params.wordGapSensitivity *
                          (otsuGap > 0.0f ? otsuGap
                                          : std::max(medianGap * 1.8f, lineHeight * 0.22f));

    switch (params.mode) {
      case GroupMode::Character:
        for (size_t i = 0; i < glyphs.size(); ++i)
          pushGroup(glyphs[i].bbox, glyphs[i].sx1, glyphs[i].sx2, int(li), int(i),
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
        pushGroup(box, lo, hi, int(li), 0, std::move(labels));
        break;
      }

      case GroupMode::Word: {
        RectI box = glyphs[0].bbox;
        float lo = glyphs[0].sx1, hi = glyphs[0].sx2;
        std::vector<int> labels = glyphs[0].labels;
        int idx = 0;
        for (size_t i = 1; i < glyphs.size(); ++i) {
          const float gap = glyphs[i].sx1 - glyphs[i - 1].sx2;
          if (gap > breakAt) {
            pushGroup(box, lo, hi, int(li), idx++, std::move(labels));
            box = glyphs[i].bbox;
            lo = glyphs[i].sx1;
            hi = glyphs[i].sx2;
            labels = glyphs[i].labels;
          } else {
            box.unionWith(glyphs[i].bbox);
            lo = std::min(lo, glyphs[i].sx1);
            hi = std::max(hi, glyphs[i].sx2);
            labels.insert(labels.end(), glyphs[i].labels.begin(), glyphs[i].labels.end());
          }
        }
        pushGroup(box, lo, hi, int(li), idx, std::move(labels));
        break;
      }
    }
  }

  seg.labelToGroup.assign(size_t(labelCount) + 1, -1);
  for (size_t gi = 0; gi < seg.groups.size(); ++gi)
    for (int l : seg.groups[gi].labels) seg.labelToGroup[l] = int(gi);

  return seg;
}

}  // namespace rta
