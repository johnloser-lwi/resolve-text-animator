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

namespace rta {

struct WarningState {
  // The host reports the clip reaching back before its source start, so the
  // reveal will begin later than the clip's first frame.
  bool sourceOffset = false;
  // An animation does not have room to finish inside the clip.
  bool clipTooShort = false;

  bool any() const { return sourceOffset || clipTooShort; }
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

}  // namespace rta
