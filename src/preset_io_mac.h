// The macOS halves of askSavePath / askOpenPath, kept out of preset_io.cpp
// because they are Objective-C++ (preset_io_mac.mm) and only build on Apple.
#pragma once

#include <string>

namespace rta {

// `dir` is where the dialog opens; empty lets the system choose. Both return an
// empty string on cancel, exactly like the Windows dialogs.
std::string macAskSavePath(const std::string& dir, const std::string& fileName);
std::string macAskOpenPath(const std::string& dir);

}  // namespace rta
