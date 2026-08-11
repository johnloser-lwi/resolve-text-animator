#include "preset.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace rta {
namespace {

// Values are written with enough precision to survive a round trip exactly.
// %.17g is the shortest form guaranteed to reproduce a double.
std::string num(double v) {
  char buf[40];
  std::snprintf(buf, sizeof(buf), "%.17g", v);
  return buf;
}

std::string escape(const std::string& in) {
  std::string out;
  for (char c : in) {
    if (c == '\\' || c == '"') {
      out += '\\';
      out += c;
    } else if (c == '\n') {
      out += "\\n";
    } else {
      out += c;
    }
  }
  return out;
}

bool unescape(const std::string& in, std::string* out) {
  out->clear();
  for (size_t i = 0; i < in.size(); ++i) {
    if (in[i] != '\\') {
      *out += in[i];
      continue;
    }
    if (++i >= in.size()) return false;  // trailing backslash: malformed
    if (in[i] == 'n')
      *out += '\n';
    else
      *out += in[i];
  }
  return true;
}

std::string trim(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\r\n");
  if (a == std::string::npos) return "";
  size_t b = s.find_last_not_of(" \t\r\n");
  return s.substr(a, b - a + 1);
}

}  // namespace

std::string writePreset(const PresetFile& p) {
  std::ostringstream o;
  o << "TextAnimatorPreset " << p.schemaVersion << "\n";
  o << "name \"" << escape(p.name) << "\"\n";

  for (const auto& kv : p.params) {
    const PresetValue& v = kv.second;
    o << kv.first << ' ' << v.kind << ' ';
    switch (v.kind) {
      case kPKBool:
        o << (v.b ? 1 : 0);
        break;
      case kPKString:
        o << '"' << escape(v.s) << '"';
        break;
      default:
        for (int i = 0; i < v.count; ++i) o << (i ? " " : "") << num(v.n[i]);
        break;
    }
    o << '\n';
  }
  return o.str();
}

bool readPreset(const std::string& text, PresetFile* out) {
  if (!out) return false;

  PresetFile got;  // built aside, so a failure part-way leaves `out` alone
  std::istringstream in(text);
  std::string line;

  if (!std::getline(in, line)) return false;
  {
    std::istringstream hdr(trim(line));
    std::string magic;
    hdr >> magic >> got.schemaVersion;
    if (magic != "TextAnimatorPreset") return false;
    if (got.schemaVersion > kPresetSchemaVersion) return false;  // written by a newer build
  }

  while (std::getline(in, line)) {
    const std::string t = trim(line);
    if (t.empty()) continue;

    const size_t sp = t.find(' ');
    if (sp == std::string::npos) return false;
    const std::string key = t.substr(0, sp);
    std::string rest = trim(t.substr(sp + 1));

    if (key == "name") {
      if (rest.size() < 2 || rest.front() != '"' || rest.back() != '"') return false;
      if (!unescape(rest.substr(1, rest.size() - 2), &got.name)) return false;
      continue;
    }

    const size_t sp2 = rest.find(' ');
    if (sp2 == std::string::npos) return false;
    PresetValue v;
    v.kind = std::atoi(rest.substr(0, sp2).c_str());
    rest = trim(rest.substr(sp2 + 1));

    switch (v.kind) {
      case kPKBool:
        v.b = std::atoi(rest.c_str()) != 0;
        break;
      case kPKString: {
        if (rest.size() < 2 || rest.front() != '"' || rest.back() != '"') return false;
        if (!unescape(rest.substr(1, rest.size() - 2), &v.s)) return false;
        break;
      }
      case kPKDouble:
      case kPKInt:
      case kPKChoice:
      case kPKRGBA: {
        std::istringstream vals(rest);
        while (v.count < 4 && vals >> v.n[v.count]) ++v.count;
        if (v.count == 0) return false;
        break;
      }
      default:
        return false;  // unknown kind: refuse rather than guess
    }
    got.params[key] = v;
  }

  *out = got;
  return true;
}

}  // namespace rta
