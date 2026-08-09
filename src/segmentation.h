// Recovers text structure from rendered pixels alone.
//
// The pipeline is: alpha mask -> connected components -> glyphs -> lines ->
// words. No text metadata is involved anywhere; every grouping decision comes
// from measuring pixel distances, which is what makes this work on any title
// generator that produces an alpha channel.
#pragma once

#include <cstdint>
#include <utility>
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

  // Italic compensation. Letters are measured in a sheared coordinate system,
  // x' = x - (y * tan(slant)), so an oblique face is grouped as though it were
  // upright. Only the MEASUREMENT is deskewed -- the pixels are never moved, so
  // groups still render exactly where the glyphs are.
  //
  // Without this an italic 'T' leans over the letter after it, which reads as
  // heavy x-overlap and chains a whole line into one glyph, and the gaps that
  // separate words shrink to nothing.
  // Manual overrides, addressed by CHARACTER index.
  //
  // Grouping is a choice of boundaries between characters, so both overrides are
  // edits to that set: mergeAt removes the boundary before a character, joining
  // it to what precedes it; splitBefore adds one, starting a new unit there.
  //
  // Character indices are the right currency because they do not move. Numbering
  // by unit meant every merge renumbered the units after it, so the second
  // correction to a line could not be expressed at all -- fix ANIMATION and the
  // number for FOR has already changed. Characters are also what a split has to
  // name anyway, so one scheme serves both.
  //
  // Set Group Mode to Character to read these numbers off the viewer.
  std::vector<int> mergeAt;
  std::vector<int> splitBefore;

  bool autoSlant = true;         // measure the slant from the image
  float italicSlant = 0.0f;      // degrees, used when autoSlant is off

  bool operator==(const DetectParams& o) const {
    return alphaThreshold == o.alphaThreshold && minBlobArea == o.minBlobArea &&
           wordGapSensitivity == o.wordGapSensitivity &&
           bridgeRadius == o.bridgeRadius && mode == o.mode &&
           autoSlant == o.autoSlant && italicSlant == o.italicSlant &&
           mergeAt == o.mergeAt && splitBefore == o.splitBefore;
  }
  bool operator!=(const DetectParams& o) const { return !(*this == o); }
};

// One animatable unit: a character, a word, or a whole line depending on mode.
struct Group {
  RectI bbox;
  // Deskewed horizontal extent. bbox is the axis-aligned box the compositor
  // works in; this pair plus Segmentation::slantTan is the PARALLELOGRAM the
  // group really occupies, which is what the diagnostics overlay draws so an
  // italic word is outlined by its own shape rather than by a rectangle that
  // swallows the letters either side of it.
  float sx1 = 0.0f, sx2 = 0.0f;
  // Glyphs merged into this group. 1 per group in Character mode; the letter
  // count of the word or line otherwise. Lets the reveal spend time in
  // proportion to how much text a group actually holds.
  int glyphCount = 1;
  // First glyph of this group within its line, for manual splits.
  int firstGlyph = 0;
  // Left edge of each glyph in this group, in image pixels. The viewer overlay
  // needs these to turn a click into "split after the n-th letter" -- letters
  // are not evenly spaced, so the position cannot be interpolated from the box.
  std::vector<int> glyphStarts;
  // Range of ORIGINAL (pre-override) unit indices this group covers, and which
  // original unit owns each glyph.
  //
  // Every override names an original index, so the label on screen must be the
  // original index too. Numbering the final groups instead renumbers them after
  // each merge, and the next index the user reads no longer means what the
  // parameter expects -- merge 4 and the old 5 becomes 5, so a second merge
  // cannot be expressed at all.
  // Character index this unit starts at, and the index of each of its
  // characters. These are the numbers the overrides speak, and the number Show
  // Detection paints -- stable whatever the grouping turns out to be.
  int glyphIndex = 0;
  std::vector<int> glyphIndices;
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
  // Italic slant actually used. Degrees for display; the tangent is what the
  // geometry needs, and deriving one from the other in two places invites them
  // to disagree. Shear at row y is (y - height/2) * slantTan.
  float slantDegrees = 0.0f;
  float slantTan = 0.0f;

  bool empty() const { return groups.empty(); }
};

// Runs the full pipeline. Returns an empty Segmentation if nothing was found.
Segmentation segment(const ImageView& src, const DetectParams& params);

// Cheap content fingerprint of the alpha channel, used to decide whether a
// cached Segmentation is still valid. Subsamples, so it is O(pixels/64).
uint64_t hashAlpha(const ImageView& src, float threshold);

}  // namespace rta
