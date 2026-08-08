// A tiny diagnostic log.
//
// Host time and clip-extent behaviour cannot be reasoned about from the outside
// -- every route to "where does this clip start" is host-specific and at least
// one of them returns a sentinel. When the timing looks wrong the only way to
// tell which is happening is to record what the host actually said.
//
// Quiet by default: nothing is written unless a value changes, so it stays
// silent during normal playback.
#pragma once

#include <cstdio>
#include <fstream>
#include <map>
#include <mutex>
#include <string>

#ifdef _WIN32
#include <cstdlib>
#endif

namespace rta {
namespace detail {

inline std::string probePath() {
#ifdef _WIN32
  const char* base = std::getenv("LOCALAPPDATA");
  if (base && *base) return std::string(base) + "\\TextAnimator\\probe.log";
  return "TextAnimator-probe.log";
#else
  const char* home = std::getenv("HOME");
  if (home && *home) return std::string(home) + "/TextAnimator-probe.log";
  return "TextAnimator-probe.log";
#endif
}

inline std::mutex& probeMutex() {
  static std::mutex m;
  return m;
}

}  // namespace detail

// Appends one line. Creates the directory lazily on first use.
inline void probeLog(const std::string& line) {
  std::lock_guard<std::mutex> lock(detail::probeMutex());
  static bool ready = false;
  static std::string path;
  if (!ready) {
    ready = true;
    path = detail::probePath();
#ifdef _WIN32
    const size_t slash = path.find_last_of('\\');
    if (slash != std::string::npos) {
      const std::string dir = path.substr(0, slash);
      std::string cmd = "cmd /c if not exist \"" + dir + "\" mkdir \"" + dir + "\" >nul 2>&1";
      std::system(cmd.c_str());
    }
#endif
  }
  std::ofstream f(path, std::ios::app);
  if (f) f << line << "\n";
}

// Logs only when `value` differs from the last value seen for `key`, so a
// per-frame call costs a string compare once the state is steady.
inline void probeOnChange(const char* key, const std::string& value) {
  // Per key. A single shared "last value" meant two alternating keys each saw
  // the other's value and both logged every time, which is noise dressed up as
  // change.
  static std::mutex m;
  static std::map<std::string, std::string> last;
  {
    std::lock_guard<std::mutex> lock(m);
    auto it = last.find(key);
    if (it != last.end() && it->second == value) return;
    last[key] = value;
  }
  probeLog(std::string(key) + ": " + value);
}

}  // namespace rta
