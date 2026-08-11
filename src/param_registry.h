// Every parameter the plugin defines, recorded as it is defined.
//
// The alternative -- what MultiTransform does -- is a struct mirroring each
// parameter by hand, written out three times: once in the struct, once in the
// serialiser, once in the parser. Adding a parameter then means three edits, and
// missing one produces a preset that silently fails to save that setting. No
// error, no symptom, until someone reloads and finds their work reverted.
//
// This plugin gained a dozen parameters in a single sitting, and one of them --
// Character Gap -- shipped defined and fetched but never read, with every test
// passing. A hand-written mirror invites exactly that class of omission.
//
// Registering name and kind at the point of definition means presets cover every
// parameter automatically, including ones added later, and there is one list
// that cannot fall out of step with itself.
#pragma once

#include <string>
#include <vector>

namespace rta {

enum ParamKind {
  kPKNone = 0,  // groups, pages, push buttons: nothing to store
  kPKDouble,
  kPKInt,
  kPKBool,
  kPKChoice,
  kPKRGBA,
  kPKString,
};

struct ParamEntry {
  std::string name;
  int kind = kPKNone;
};

// Function-local static rather than namespace scope: describeInContext runs
// while the host is loading the plugin, which is when static initialisation
// order is least predictable.
inline std::vector<ParamEntry>& paramRegistry() {
  static std::vector<ParamEntry> entries;
  return entries;
}

inline void registerParam(const std::string& name, int kind) {
  if (kind == kPKNone || name.empty()) return;
  // describeInContext can run more than once, once per supported context, and
  // the registry must not accumulate duplicates or a preset would write every
  // value several times over.
  for (const ParamEntry& e : paramRegistry())
    if (e.name == name) return;
  paramRegistry().push_back(ParamEntry{name, kind});
}

}  // namespace rta
