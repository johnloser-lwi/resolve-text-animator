// Warnings the renderer discovers, read by the viewer overlay.
//
// The Inspector is a poor place for these: Resolve gives a plugin no control
// over a parameter's colour, truncates its label to about half the row, and
// draws a value field beside it whether or not one makes sense. The overlay has
// the draw suite, so it can put real red text over the picture where it will
// actually be noticed.
//
// The two live in different translation units and the render side only learns
// some of this after segmenting, so it is passed through a small registry keyed
// on the effect instance rather than squeezed into a parameter.
#pragma once

#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace rta {

struct WarningState {
  // The host reports the clip reaching back before its source start, so the
  // reveal will begin later than the clip's first frame.
  bool sourceOffset = false;
  // An animation does not have room to finish inside the clip.
  bool clipTooShort = false;
  // The reference text could not be reconciled with the pixels, so automatic
  // detection was used instead. A warning rather than a status line because the
  // fallback is invisible otherwise -- the result would simply look like the
  // string had been ignored.
  bool referenceMismatch = false;
  std::string referenceMessage;

  bool any() const { return sourceOffset || clipTooShort || referenceMismatch; }
};

namespace detail {
inline std::mutex& warnMutex() {
  static std::mutex m;
  return m;
}
inline std::map<const void*, WarningState>& warnMap() {
  static std::map<const void*, WarningState> m;
  return m;
}
}  // namespace detail

inline void setWarningState(const void* effect, const WarningState& s) {
  std::lock_guard<std::mutex> lock(detail::warnMutex());
  detail::warnMap()[effect] = s;
}

inline WarningState warningState(const void* effect) {
  std::lock_guard<std::mutex> lock(detail::warnMutex());
  auto it = detail::warnMap().find(effect);
  return it == detail::warnMap().end() ? WarningState{} : it->second;
}

inline void clearWarningState(const void* effect) {
  std::lock_guard<std::mutex> lock(detail::warnMutex());
  detail::warnMap().erase(effect);
}

// What the analysis measured, for the overlay to display and for the Auto
// Italic toggle to hand to the manual control.
//
// Kept separate from WarningState because it is not a warning -- any() must go
// on meaning "something is wrong", or the red alert text would fire on every
// clip that merely contains slanted type.
// One detected unit, as the viewer needs to see it.
//
// Positions are IMAGE PIXELS, top-down, alongside the image size they were
// measured at. The overlay works in canonical coordinates with y up, so it
// rescales these against the region of definition -- which keeps the mapping
// correct at any render scale without the overlay having to know what that
// scale is.
struct DetectedUnit {
  int x1 = 0, y1 = 0, x2 = 0, y2 = 0;
  // Deskewed extent, so the overlay can outline an italic unit with the
  // parallelogram it really occupies rather than an upright box.
  float sx1 = 0.0f, sx2 = 0.0f;
  // Character index this unit starts at, and the index of each of its letters.
  // Overrides are addressed by character, so these are what the viewer writes.
  int glyphIndex = 0;
  std::vector<int> glyphStarts;   // left edge of each letter, image pixels
  std::vector<int> glyphIndices;  // character index of each letter
};

struct AnalysisState {
  float slantDegrees = 0.0f;
  bool haveSlant = false;

  int width = 0, height = 0;  // image the units were measured in
  std::vector<DetectedUnit> units;
};

namespace detail {
inline std::map<const void*, AnalysisState>& analysisMap() {
  static std::map<const void*, AnalysisState> m;
  return m;
}
}  // namespace detail

inline void setAnalysisState(const void* effect, const AnalysisState& s) {
  std::lock_guard<std::mutex> lock(detail::warnMutex());
  detail::analysisMap()[effect] = s;
}

inline AnalysisState analysisState(const void* effect) {
  std::lock_guard<std::mutex> lock(detail::warnMutex());
  auto it = detail::analysisMap().find(effect);
  return it == detail::analysisMap().end() ? AnalysisState{} : it->second;
}

inline void clearAnalysisState(const void* effect) {
  std::lock_guard<std::mutex> lock(detail::warnMutex());
  detail::analysisMap().erase(effect);
}

}  // namespace rta
