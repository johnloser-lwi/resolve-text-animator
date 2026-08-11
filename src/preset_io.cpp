#include "preset_io.h"

#include <fstream>
#include <sstream>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <commdlg.h>
#include <shlobj.h>
#endif

namespace rta {
namespace {

#ifdef _WIN32
// Creates every missing level of `path`. CreateDirectory only makes the last
// one, so a fresh machine with no Documents\TextAnimator would fail on the
// first save.
//
// Deliberately the API and not a shell-out. An earlier diagnostic log built its
// folder with std::system("cmd /c mkdir"), and that pops a console window every
// time the plugin loads -- visible, and reported as a bug.
void makeDirs(const std::string& path) {
  for (size_t i = 0; i <= path.size(); ++i) {
    if (i == path.size() || path[i] == '\\' || path[i] == '/') {
      if (i < 3) continue;  // skip the drive root
      CreateDirectoryA(path.substr(0, i).c_str(), nullptr);
    }
  }
}
#endif

}  // namespace

std::string presetFolder() {
#ifdef _WIN32
  char docs[MAX_PATH] = {0};
  std::string base;
  if (SUCCEEDED(SHGetFolderPathA(nullptr, CSIDL_PERSONAL, nullptr, 0, docs)) && docs[0]) {
    base = docs;
  } else if (const char* local = std::getenv("LOCALAPPDATA")) {
    base = local;
  } else {
    return "";
  }
  const std::string dir = base + "\\TextAnimator\\Presets";
  makeDirs(dir);
  return dir;
#else
  const char* home = std::getenv("HOME");
  return home ? std::string(home) + "/TextAnimator/Presets" : std::string();
#endif
}

#ifdef _WIN32
namespace {

// Shared setup for both dialogs. The filter is a double-NUL terminated list,
// which is why it cannot be a plain string literal.
OPENFILENAMEA baseDialog(char* buf, size_t bufLen, const std::string& dir, const char* filter) {
  OPENFILENAMEA ofn;
  ZeroMemory(&ofn, sizeof(ofn));
  ofn.lStructSize = sizeof(ofn);
  ofn.lpstrFilter = filter;
  ofn.lpstrFile = buf;
  ofn.nMaxFile = DWORD(bufLen);
  ofn.lpstrInitialDir = dir.empty() ? nullptr : dir.c_str();
  ofn.lpstrDefExt = "tapreset";
  return ofn;
}

}  // namespace
#endif

std::string askSavePath(const std::string& suggestedName) {
#ifdef _WIN32
  static const char kFilter[] = "Text Animator preset\0*.tapreset\0All files\0*.*\0\0";
  char buf[MAX_PATH] = {0};
  const std::string safe = suggestedName.empty() ? "preset" : suggestedName;
  strncpy_s(buf, safe.c_str(), _TRUNCATE);

  OPENFILENAMEA ofn = baseDialog(buf, sizeof(buf), presetFolder(), kFilter);
  ofn.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
  return GetSaveFileNameA(&ofn) ? std::string(buf) : std::string();
#else
  (void)suggestedName;
  return "";
#endif
}

std::string askOpenPath() {
#ifdef _WIN32
  static const char kFilter[] = "Text Animator preset\0*.tapreset\0All files\0*.*\0\0";
  char buf[MAX_PATH] = {0};
  OPENFILENAMEA ofn = baseDialog(buf, sizeof(buf), presetFolder(), kFilter);
  ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
  return GetOpenFileNameA(&ofn) ? std::string(buf) : std::string();
#else
  return "";
#endif
}

bool writeFile(const std::string& path, const std::string& text) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) return false;
  f << text;
  return bool(f);
}

bool readFile(const std::string& path, std::string* text) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  std::ostringstream o;
  o << f.rdbuf();
  *text = o.str();
  return true;
}

std::string stemOf(const std::string& path) {
  size_t a = path.find_last_of("\\/");
  a = (a == std::string::npos) ? 0 : a + 1;
  const size_t b = path.find_last_of('.');
  return (b != std::string::npos && b > a) ? path.substr(a, b - a) : path.substr(a);
}

}  // namespace rta
