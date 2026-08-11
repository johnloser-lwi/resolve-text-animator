// Preset schema, serialiser and parser.
//
// Deliberately free of any OFX or Win32 dependency, so the format can be
// exercised in the harness like the rest of the pipeline. The file dialogs live
// in preset_io.h, which is the only part that needs a desktop.
//
// The parser is strict on purpose: a preset that loads the wrong values, or half
// a file, is worse than one that refuses. Any failure leaves the output
// untouched.
#pragma once

#include <map>
#include <string>

#include "param_registry.h"

namespace rta {

// Bumped only on an incompatible change. Unknown keys are ignored and missing
// ones fall back to the parameter's default, so adding a parameter needs no
// bump -- which is the point of driving the format off the registry.
constexpr int kPresetSchemaVersion = 1;

// One parameter's value, in whichever shape its kind needs.
struct PresetValue {
  int kind = kPKNone;
  double n[4] = {0.0, 0.0, 0.0, 0.0};
  int count = 0;  // how many of n are meaningful
  bool b = false;
  std::string s;
};

using PresetMap = std::map<std::string, PresetValue>;

struct PresetFile {
  int schemaVersion = kPresetSchemaVersion;
  std::string name;
  PresetMap params;
};

// Serialise to the on-disk text form.
std::string writePreset(const PresetFile& p);

// Parse. Returns false and leaves `out` untouched on any malformed input.
bool readPreset(const std::string& text, PresetFile* out);

}  // namespace rta
