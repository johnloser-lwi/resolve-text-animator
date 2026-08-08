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
  int area = 0;
  bool alive = false;
};

// Fraction of the narrower box's width that the two boxes share in x.
float xOverlapRatio(const RectI& a, const RectI& b) {
  const int overlap = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
  if (overlap <= 0) return 0.0f;
  return float(overlap) / float(std::max(1, std::min(a.width(), b.width())));
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
bool sameGlyph(const RectI& a, const RectI& b) {
  return xOverlapRatio(a, b) >= 0.6f && yOverlapRatio(a, b) <= 0.2f;
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
  std::vector<Component> comps(size_t(labelCount) + 1);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const size_t p = size_t(y) * w + x;
      if (!mask[p] || prov[p] == 0) continue;
      const int lab = remap[uf.find(prov[p])];
      seg.labelImage[p] = lab;
      Component& c = comps[lab];
      if (c.area == 0) {
        c.bbox = RectI{x, y, x + 1, y + 1};
      } else {
        c.bbox.x1 = std::min(c.bbox.x1, x);
        c.bbox.y1 = std::min(c.bbox.y1, y);
        c.bbox.x2 = std::max(c.bbox.x2, x + 1);
        c.bbox.y2 = std::max(c.bbox.y2, y + 1);
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
        if (sameGlyph(comps[ids[i]].bbox, comps[ids[j]].bbox)) guf.unite(int(i), int(j));
      }
    }
    std::vector<int> rootToGlyph(ids.size(), -1);
    auto& glyphs = lineGlyphs[li];
    for (size_t i = 0; i < ids.size(); ++i) {
      const int root = guf.find(int(i));
      if (rootToGlyph[root] < 0) {
        rootToGlyph[root] = int(glyphs.size());
        glyphs.push_back(Glyph{comps[ids[i]].bbox, {}});
      }
      Glyph& g = glyphs[rootToGlyph[root]];
      g.bbox.unionWith(comps[ids[i]].bbox);
      g.labels.push_back(ids[i]);
    }
    std::sort(glyphs.begin(), glyphs.end(),
              [](const Glyph& a, const Glyph& b) { return a.bbox.x1 < b.bbox.x1; });
  }

  // 8. Word breaks from gap statistics. The median gap adapts to the font's
  //    tracking; the line-height floor keeps tightly-set text from shattering.
  std::vector<int> allGaps;
  for (const auto& glyphs : lineGlyphs) {
    for (size_t i = 1; i < glyphs.size(); ++i)
      allGaps.push_back(std::max(0, glyphs[i].bbox.x1 - glyphs[i - 1].bbox.x2));
  }
  float medianGap = 0.0f;
  if (!allGaps.empty()) {
    std::nth_element(allGaps.begin(), allGaps.begin() + allGaps.size() / 2, allGaps.end());
    medianGap = float(allGaps[allGaps.size() / 2]);
  }

  // 9. Emit groups.
  auto pushGroup = [&](const RectI& bbox, int line, int idx, std::vector<int> labels) {
    Group g;
    g.bbox = bbox;
    g.line = line;
    g.indexInLine = idx;
    g.labels = std::move(labels);
    seg.groups.push_back(std::move(g));
  };

  for (size_t li = 0; li < lineGlyphs.size(); ++li) {
    const auto& glyphs = lineGlyphs[li];
    if (glyphs.empty()) continue;
    const float lineHeight = float(lineSpans[li].height());
    const float breakAt =
        params.wordGapSensitivity * std::max(medianGap * 1.8f, lineHeight * 0.22f);

    switch (params.mode) {
      case GroupMode::Character:
        for (size_t i = 0; i < glyphs.size(); ++i)
          pushGroup(glyphs[i].bbox, int(li), int(i), glyphs[i].labels);
        break;

      case GroupMode::Line: {
        RectI box;
        std::vector<int> labels;
        for (const auto& g : glyphs) {
          box.unionWith(g.bbox);
          labels.insert(labels.end(), g.labels.begin(), g.labels.end());
        }
        pushGroup(box, int(li), 0, std::move(labels));
        break;
      }

      case GroupMode::Word: {
        RectI box = glyphs[0].bbox;
        std::vector<int> labels = glyphs[0].labels;
        int idx = 0;
        for (size_t i = 1; i < glyphs.size(); ++i) {
          const float gap = float(glyphs[i].bbox.x1 - glyphs[i - 1].bbox.x2);
          if (gap > breakAt) {
            pushGroup(box, int(li), idx++, std::move(labels));
            box = glyphs[i].bbox;
            labels = glyphs[i].labels;
          } else {
            box.unionWith(glyphs[i].bbox);
            labels.insert(labels.end(), glyphs[i].labels.begin(), glyphs[i].labels.end());
          }
        }
        pushGroup(box, int(li), idx, std::move(labels));
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
