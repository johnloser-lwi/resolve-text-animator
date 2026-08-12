// Recovers text structure from rendered pixels alone.
//
// The pipeline is: alpha mask -> connected components -> glyphs -> lines ->
// words. By default no text metadata is involved anywhere; every grouping
// decision comes from measuring pixel distances, which is what makes this work
// on any title generator that produces an alpha channel.
//
// The one exception is opt-in: supply DetectParams::referenceText and the string
// decides the character and word boundaries instead of the gap statistics. Left
// empty -- which it is by default -- the pipeline is exactly as described above.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "image_view.h"

namespace rta {

enum class GroupMode { Character = 0, Word = 1, Line = 2 };

struct DetectParams {
  float alphaThreshold = 0.15f;
  float wordGapSensitivity = 1.0f;

  // Despeckling and bridging, as FRACTIONS OF LETTER HEIGHT.
  //
  // The pixel-denominated pair below them means something different at every
  // resolution: the same Character Padding of 3px gave 9 units at full res and 4
  // at half on the same title, so a proxy render regrouped the whole frame. Every
  // other threshold here is already relative for exactly this reason; these two
  // were the last absolute ones.
  //
  // The yardstick is the median LINE height, measured from the ink profile before
  // anything is labelled -- so it is available before bridging, which has to
  // happen first, and it tracks proxy scale, timeline resolution and point size
  // together the way letter height does.
  //
  // charPadding is a radius, so it scales with height; minMarkArea is an area, so
  // it scales with height SQUARED.
  //
  // minMarkArea's default is what the old 4-pixel despeckle came to on a title of
  // ordinary size, so nothing changes at the resolution it was tuned at -- it now
  // simply follows the type everywhere else, which the pixel value never did.
  float charPadding = 0.0f;
  float minMarkArea = 0.0005f;

  // Legacy pixel padding. Kept, and ADDED to charPadding, because a project saved
  // before the change carries a pixel value and no host signal says it is old --
  // OFX restores an absent parameter to its default, so a version flag cannot
  // tell a fresh instance from an old save. Its default was zero, so preserving
  // it costs nothing: almost no project has one, and the few that do open
  // grouping exactly as they did.
  //
  // The old pixel DESPECKLE is not kept the same way, and deliberately. Its
  // default was 4, so every existing project carries a value -- honouring them
  // would leave the pixel dependency in place for everyone, which is the whole
  // thing being removed here.
  int bridgeRadius = 0;          // dilate before labeling, to join script fonts

  // The widest gap that still counts as being INSIDE a word, as a FRACTION OF
  // LETTER HEIGHT. Anything wider starts a new word.
  //
  // Relative, not pixels. A pixel value silently changes meaning the moment the
  // proxy resolution changes, or the timeline resolution, or the point size --
  // the same title at half resolution has half the gaps, so a threshold tuned at
  // full res splits every letter in proxy. Letter height moves with all three,
  // so a ratio holds. It is also the same yardstick the automatic path uses.
  //
  // A word space is roughly a quarter to a third of letter height in most
  // typefaces, so useful values sit near 0.3.
  //
  // Unlike bridging, the letters are never fused: every character boundary
  // survives, so Character mode still shows letters and a manual split inside a
  // word is still possible.
  //
  // Zero leaves word breaks to the measured gap statistics.
  float maxLetterGap = 0.0f;
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

  // The text the title actually says. Empty means automatic detection, and with
  // it empty NOT ONE branch of the pipeline behaves differently -- this is an
  // extra layer of protection over a detector that is already good, never a
  // replacement for it.
  //
  // Supplied, it is AUTHORITATIVE rather than advisory. The marks on each line
  // are merged down until their count equals that line's character count, and
  // the word cuts come from the string's spaces; no gap statistic decides
  // anything. Advisory was tried first -- the string supplied word counts while
  // the gaps still placed the cuts -- and improved nothing.
  //
  // Whitespace only separates words. Line breaks come from the PIXELS: text
  // wraps at word boundaries, so words pack onto the detected lines in reading
  // order, and a newline in the string means no more than a space.
  std::string referenceText;

  bool autoSlant = true;         // measure the slant from the image
  float italicSlant = 0.0f;      // degrees, used when autoSlant is off

  bool operator==(const DetectParams& o) const {
    return alphaThreshold == o.alphaThreshold &&
           wordGapSensitivity == o.wordGapSensitivity && charPadding == o.charPadding &&
           minMarkArea == o.minMarkArea &&
           bridgeRadius == o.bridgeRadius && maxLetterGap == o.maxLetterGap && mode == o.mode &&
           autoSlant == o.autoSlant && italicSlant == o.italicSlant &&
           mergeAt == o.mergeAt && splitBefore == o.splitBefore &&
           referenceText == o.referenceText;
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

// What the reference text did, if anything.
//
// Failed matters as much as Applied: the pixels and the string disagreed, the
// automatic result is what came out, and the user has to be TOLD that -- a
// silent fallback would look like the string was honoured.
enum class RefStatus { NotUsed = 0, Applied, Failed };

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

  RefStatus refStatus = RefStatus::NotUsed;
  // Why it failed, in the user's terms -- which line, and the two counts that
  // disagreed. Empty unless refStatus is Failed.
  std::string refMessage;

  bool empty() const { return groups.empty(); }
};

// Runs the full pipeline. Returns an empty Segmentation if nothing was found.
Segmentation segment(const ImageView& src, const DetectParams& params);

// Cheap content fingerprint of the alpha channel, used to decide whether a
// cached Segmentation is still valid. Subsamples, so it is O(pixels/64).
uint64_t hashAlpha(const ImageView& src, float threshold);

}  // namespace rta
