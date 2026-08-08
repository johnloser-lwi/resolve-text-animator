// Recovers text structure from rendered pixels alone.
//
// The pipeline is: alpha mask -> connected components -> glyphs -> lines ->
// words. No text metadata is involved anywhere; every grouping decision comes
// from measuring pixel distances, which is what makes this work on any title
// generator that produces an alpha channel.
#pragma once

#include <cstdint>
#include <vector>

#include "image_view.h"

namespace rta {

enum class GroupMode { Character = 0, Word = 1, Line = 2 };

struct DetectParams {
  float alphaThreshold = 0.15f;
  int minBlobArea = 4;           // in pixels, at analysis resolution
  float wordGapSensitivity = 1.0f;
  int bridgeRadius = 0;          // dilate before labeling, to join script fonts
  GroupMode mode = GroupMode::Word;

  bool operator==(const DetectParams& o) const {
    return alphaThreshold == o.alphaThreshold && minBlobArea == o.minBlobArea &&
           wordGapSensitivity == o.wordGapSensitivity &&
           bridgeRadius == o.bridgeRadius && mode == o.mode;
  }
  bool operator!=(const DetectParams& o) const { return !(*this == o); }
};

// One animatable unit: a character, a word, or a whole line depending on mode.
struct Group {
  RectI bbox;
  int line = 0;        // 0-based, top to bottom
  int indexInLine = 0; // 0-based, left to right
  std::vector<int> labels;  // component labels belonging to this group
};

struct Segmentation {
  int width = 0;
  int height = 0;
  // Per-pixel component label; 0 means background, >0 indexes a component.
  // Groups own labels, so this doubles as the per-group stencil.
  std::vector<int32_t> labelImage;
  // Maps a component label to the index of the group that owns it, or -1.
  std::vector<int> labelToGroup;
  std::vector<Group> groups;
  int lineCount = 0;

  bool empty() const { return groups.empty(); }
};

// Runs the full pipeline. Returns an empty Segmentation if nothing was found.
Segmentation segment(const ImageView& src, const DetectParams& params);

// Cheap content fingerprint of the alpha channel, used to decide whether a
// cached Segmentation is still valid. Subsamples, so it is O(pixels/64).
uint64_t hashAlpha(const ImageView& src, float threshold);

}  // namespace rta
