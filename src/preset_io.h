// Where preset files live, and the dialogs that find them.
//
// Separated from preset.cpp because everything here is Win32 and none of it can
// be exercised without a desktop. Keeping the schema and the parser free of that
// is what lets them be tested offline.
#pragma once

#include <string>

namespace rta {

// Documents\TextAnimator\Presets, created on demand -- somewhere the user can
// browse to, copy from and back up. Falls back to LocalAppData if Documents
// cannot be resolved.
std::string presetFolder();

// Native Save/Open dialogs. Return an empty string if the user cancels, which
// is a normal outcome and never an error.
std::string askSavePath(const std::string& suggestedName);
std::string askOpenPath();

bool writeFile(const std::string& path, const std::string& text);
bool readFile(const std::string& path, std::string* text);

// Filename without directory or extension, for naming a preset after its file.
std::string stemOf(const std::string& path);

}  // namespace rta
