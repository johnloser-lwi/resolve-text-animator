#include "preset_io_mac.h"

#import <AppKit/AppKit.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>

#include <cstdio>

namespace rta {
namespace {

// AppKit panels may only be run on the main thread. A host that calls
// changedParam from a worker thread cannot be served by dispatching to the main
// queue either: if the main thread is the one waiting on us, that deadlocks the
// host. osascript runs the same standard dialog in its own process, so it is
// safe from any thread -- slower to appear, which is the price of not hanging.
std::string runAppleScript(const std::string& script) {
  // Single-quoted for /bin/sh; a ' inside becomes '\''.
  std::string quoted = "'";
  for (char ch : script) quoted += (ch == '\'') ? std::string("'\\''") : std::string(1, ch);
  quoted += "'";
  FILE* p = popen(("/usr/bin/osascript -e " + quoted + " 2>/dev/null").c_str(), "r");
  if (!p) return "";
  std::string out;
  char buf[512];
  while (fgets(buf, sizeof(buf), p)) out += buf;
  pclose(p);  // a cancel exits non-zero with nothing printed, which is "" below
  while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
  return out;
}

// An AppleScript string literal: backslashes and double quotes escaped.
std::string asLiteral(const std::string& s) {
  std::string r = "\"";
  for (char ch : s) {
    if (ch == '\\' || ch == '"') r += '\\';
    r += ch;
  }
  return r + "\"";
}

std::string locationClause(const std::string& dir) {
  return dir.empty() ? std::string() : " default location (POSIX file " + asLiteral(dir) + ")";
}

UTType* presetType() {
  // An extension the system does not know still gets a dynamic type, but a nil
  // array entry would throw, so the callers check rather than trust that.
  return [UTType typeWithFilenameExtension:@"tapreset"];
}

NSString* toNS(const std::string& s) { return [NSString stringWithUTF8String:s.c_str()]; }

std::string fromURL(NSURL* url) {
  const char* p = url ? url.path.fileSystemRepresentation : nullptr;
  return p ? std::string(p) : std::string();
}

}  // namespace

std::string macAskSavePath(const std::string& dir, const std::string& fileName) {
  if (![NSThread isMainThread]) {
    std::string path =
        runAppleScript("POSIX path of (choose file name with prompt \"Save Text Animator preset\""
                       " default name " + asLiteral(fileName) + locationClause(dir) + ")");
    // Unlike NSSavePanel, choose file name keeps whatever was typed, and a file
    // without the extension would be hidden from the Load dialog.
    static const std::string kExt = ".tapreset";
    if (!path.empty() && (path.size() < kExt.size() ||
                          path.compare(path.size() - kExt.size(), kExt.size(), kExt) != 0))
      path += kExt;
    return path;
  }
  @autoreleasepool {
    NSSavePanel* panel = [NSSavePanel savePanel];
    panel.title = @"Save Text Animator preset";
    panel.nameFieldStringValue = toNS(fileName);
    panel.canCreateDirectories = YES;
    if (UTType* t = presetType()) panel.allowedContentTypes = @[ t ];
    if (!dir.empty()) panel.directoryURL = [NSURL fileURLWithPath:toNS(dir) isDirectory:YES];
    return [panel runModal] == NSModalResponseOK ? fromURL(panel.URL) : std::string();
  }
}

std::string macAskOpenPath(const std::string& dir) {
  if (![NSThread isMainThread]) {
    return runAppleScript("POSIX path of (choose file with prompt \"Load Text Animator preset\"" +
                          locationClause(dir) + ")");
  }
  @autoreleasepool {
    NSOpenPanel* panel = [NSOpenPanel openPanel];
    panel.title = @"Load Text Animator preset";
    panel.canChooseFiles = YES;
    panel.canChooseDirectories = NO;
    panel.allowsMultipleSelection = NO;
    if (UTType* t = presetType()) panel.allowedContentTypes = @[ t ];
    if (!dir.empty()) panel.directoryURL = [NSURL fileURLWithPath:toNS(dir) isDirectory:YES];
    return [panel runModal] == NSModalResponseOK ? fromURL(panel.URLs.firstObject) : std::string();
  }
}

}  // namespace rta
