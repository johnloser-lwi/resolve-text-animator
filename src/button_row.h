#pragma once

// From Multi Transform (resolve-multi-transform, src/ButtonRow.h on its
// inspector-tabs branch), where its rules were worked out; see that repo's
// docs/handoff-inspector-tabs.md for how. Only the namespace differs.

// Resolve's button row: a row of exclusive buttons in the Inspector, used for
// tabs.
//
// OFX pages are the standard tab mechanism and Resolve does not draw them: it
// reports kOfxParamHostPropMaxPages = 0 and flattens every page into one list
// (measured on Resolve 21.2). What it does have is an undocumented parameter
// type, "OfxParamTypeButtonRow" -- the row of buttons its own effects use, and
// the one other Resolve plugins build their tab bars from.
//
// Its contract, worked out from Resolve's own assertion and from a shipping
// plugin that uses it (nothing is documented):
//
//  - Per button: an id in kOfxParamPropChoiceEnum, a label in
//    kOfxParamPropChoiceOption, and EXACTLY kButtonRowIconsPerButton entries in
//    kOfxParamPropChoiceIcon. Resolve asserts
//        icons == buttons * s_ButtonRowNumIcons
//    and a mismatch aborts the whole application when the instance is created
//    -- it is not rejected, it is a crash on project load.
//  - Each "icon" is an IMAGE PATH for one button state. The Edit and Color
//    pages substitute it into a fixed style sheet (the template is a string
//    in Resolve's binary):
//        QPushButton{min-width: 26px; ... background-color: transparent;}
//        QPushButton{image: url(%1);}
//        QPushButton:hover{image: url(%1);}            ... 8 states in all
//    A text button closes the url( itself and carries on with its own
//    declarations, and reopens an url( for Resolve's closing `)` -- see
//    ButtonRowIcons, which also says what happens when that is got wrong.
//    The faces are plain background images shipped in the bundle
//    (packaging/tab-faces, made by scripts/make-tab-faces.py); the label
//    comes from qproperty-text.
//    Fusion's page reuses the description but draws its own native buttons,
//    and reads two of the eight icons as images -- see ButtonRowIcons for how
//    one set of strings serves both.
//  - The value is a STRING: the id of the selected button. Default likewise.
//
// Only Resolve knows the type -- its Edit/Color pages, and its Fusion page,
// which reuses the same description (see ButtonRowIcons). Any other host would
// refuse it, so there the same parameter is defined as an ordinary Choice with
// the same options. The accessors below hide the difference and deal in
// button indices.
//
// The C++ support library has no class for an unknown type, so this goes
// through the raw parameter and property suites.

#include <cstring>
#include <string>
#include <vector>

#include "ofxsImageEffect.h"
#include "ofxsSupportPrivate.h"

namespace rta {

constexpr const char* kOfxParamTypeButtonRow  = "OfxParamTypeButtonRow";
constexpr const char* kOfxParamPropChoiceIcon = "OfxParamPropChoiceIcon";   // Resolve-only, undocumented

/// Style sheets Resolve wants per button (one per button state). Fixed by the
/// host; a different count is a crash, see above.
constexpr int kButtonRowIconsPerButton = 8;

/// The 8 button states, in the order of Resolve's style-sheet template, with
/// the face image and text colour for each. Faces are files in the bundle's
/// Contents/Resources/tab-faces (see scripts/make-tab-faces.py).
struct ButtonRowStateStyle { const char* face; const char* text; };
constexpr ButtonRowStateStyle kButtonRowState[kButtonRowIconsPerButton] = {
    { "normal",   "#a8a8a8" },   // 0 normal
    { "hover",    "#f4f4f4" },   // 1 :hover
    { "hover",    "#f4f4f4" },   // 2 :pressed
    { "checked",  "#f4f4f4" },   // 3 :checked
    { "checked",  "#f4f4f4" },   // 4 :checked:hover
    { "checked",  "#f4f4f4" },   // 5 :checked:pressed
    { "disabled", "#6f6f6f" },   // 6 :disabled
    { "checked",  "#f4f4f4" },   // 7 :disabled:checked
};

/// Button size in the Inspector, in pixels. The face images are drawn at this
/// aspect (3x), so change scripts/make-tab-faces.py with it.
constexpr int kButtonRowWidthPx  = 56;
constexpr int kButtonRowHeightPx = 22;

/// The declarations every state of a button shares: its size, and the text
/// colour for that state. Resolve's base rule pins every button to 26 px
/// square; these override it.
inline std::string ButtonRowDecls(int state)
{
    const std::string w = std::to_string(kButtonRowWidthPx), h = std::to_string(kButtonRowHeightPx);
    return "min-width:" + w + "px;max-width:" + w + "px;"
           "min-height:" + h + "px;max-height:" + h + "px;"
           "border-radius:0px;"
           "color:" + std::string(kButtonRowState[state].text) + ";";
}

/** @brief The 8 "icons" for one button.
 *
 * The Edit and Color pages wrap each as  QPushButton:<state>{image: url(<icon>);}
 * so an icon closes that url straight away, adds the declarations that make a
 * text button, and ENDS by reopening `image:url("<face>"` -- Resolve's own
 * `);}` then closes it and the rule is valid CSS:
 *     QPushButton:checked{image: url("f.png");min-width:..;color:..;image:url("f.png");}
 * Leave the end unbalanced in any other way and Qt rejects the WHOLE sheet
 * ("Could not parse widget stylesheet" in Resolve's log, once per button) and
 * the buttons fall back to blank default boxes.
 *
 * Fusion's page reuses the same description but draws the row as its native
 * MultiButton, and it reads icons 0 and 1 of each button as IMAGES -- a style
 * sheet there is a broken image, and the buttons came out as bare text with no
 * frame. Empty icons it skips, and then it draws its ordinary bordered buttons.
 * So icons 0 and 1 (normal, hover) are left EMPTY, which the Edit page also
 * skips, and icon 2 (pressed) carries their rules for it: it closes its own
 * rule and opens the normal and hover rules after it.
 *
 * The label must be in the normal (no pseudo-state) rule: Qt applies
 * qproperty-* once, when the widget is polished, and only from such a rule.
 */
inline std::vector<std::string> ButtonRowIcons(const std::string& facesDir, const char* label)
{
    const auto face = [&](int state) {
        return "\"" + facesDir + "/" + kButtonRowState[state].face + ".png\"";
    };
    // A state's own rule body, ending in an url( left open for Resolve to close.
    const auto rule = [&](int state) {
        return face(state) + ");" + ButtonRowDecls(state) + "image:url(" + face(state);
    };

    std::vector<std::string> icons(kButtonRowIconsPerButton);
    // 0 normal and 1 hover: empty, for Fusion. Their rules ride in icon 2.
    icons[2] = face(2) + ");" + ButtonRowDecls(2) + "}"
             + "QPushButton{" + ButtonRowDecls(0)
             + "qproperty-text:\"" + label + "\";"
             + "image:url(" + face(0) + ");}"
             + "QPushButton:hover{" + ButtonRowDecls(1) + "image:url(" + face(1);
    for (int k = 3; k < kButtonRowIconsPerButton; ++k) icons[k] = rule(k);
    return icons;
}

/** @brief Where the tab faces are installed: <bundle>/Contents/Resources/tab-faces,
 *  with forward slashes (Qt style-sheet urls). Empty if the host does not say
 *  where the plugin is. */
inline std::string TabFacesDir(OFX::ImageEffectDescriptor& desc)
{
    std::string file;
    try { file = desc.getPropertySet().propGetString(kOfxPluginPropFilePath); }
    catch (...) { return std::string(); }
    for (char& c : file) if (c == '\\') c = '/';
    while (!file.empty() && file.back() == '/') file.pop_back();

    // The host may hand over the bundle itself or the binary inside it.
    const std::string marker = ".ofx.bundle";
    const std::size_t at = file.find(marker);
    if (at == std::string::npos) return std::string();
    return file.substr(0, at + marker.size()) + "/Contents/Resources/tab-faces";
}

struct TabSpec
{
    const char* id;      ///< stored value; never shown, never change it
    const char* label;   ///< button text
};

/** @brief True when the host is Resolve itself (not its Fusion page, and not
 *  any other host), which is where the button-row type exists. */
inline bool HostHasButtonRow()
{
    const OFX::ImageEffectHostDescription* h = OFX::getImageEffectHostDescription();
    return h && (h->hostName == "DaVinciResolve" || h->hostName == "DaVinciResolveLite");
}

/** @brief Define one Resolve button row. False if the host is not Resolve, the
 *  faces cannot be located, or the host refuses the type -- the caller then
 *  falls back to a Choice.
 *
 * @p defaultIndex -1 starts with no button selected (an empty value), which is
 * how the second row of a two-row tab bar starts out.
 */
inline bool DefineButtonRow(OFX::ImageEffectDescriptor& desc, const char* name, const char* hint,
                            const std::vector<TabSpec>& tabs, int defaultIndex)
{
    const std::string faces = HostHasButtonRow() ? TabFacesDir(desc) : std::string();
    if (faces.empty()) return false;

    OfxPropertySetHandle props = nullptr;
    const OfxStatus st = OFX::Private::gParamSuite->paramDefine(
        desc.getParamSetHandle(), kOfxParamTypeButtonRow, name, &props);
    if (st != kOfxStatOK || !props) return false;

    OfxPropertySuiteV1* p = OFX::Private::gPropSuite;
    p->propSetString(props, kOfxPropLabel, 0, "");     // no label column: the row is the control
    p->propSetString(props, kOfxParamPropScriptName, 0, name);
    p->propSetString(props, kOfxParamPropHint, 0, hint);

    // Navigation, not a value: nothing to keyframe, undo, save or render.
    p->propSetInt(props, kOfxParamPropAnimates, 0, 0);
    p->propSetInt(props, kOfxParamPropCanUndo, 0, 0);
    p->propSetInt(props, kOfxParamPropPersistant, 0, 0);
    p->propSetInt(props, kOfxParamPropEvaluateOnChange, 0, 0);

    const int n = static_cast<int>(tabs.size());
    for (int i = 0; i < n; ++i)
    {
        p->propSetString(props, kOfxParamPropChoiceEnum, i, tabs[i].id);
        p->propSetString(props, kOfxParamPropChoiceOption, i, tabs[i].label);
        const std::vector<std::string> icons = ButtonRowIcons(faces, tabs[i].label);
        for (int k = 0; k < kButtonRowIconsPerButton; ++k)
            p->propSetString(props, kOfxParamPropChoiceIcon,
                             i * kButtonRowIconsPerButton + k, icons[k].c_str());
    }
    const bool none = defaultIndex < 0 || defaultIndex >= n;
    p->propSetString(props, kOfxParamPropDefault, 0, none ? "" : tabs[defaultIndex].id);
    return true;
}

/** @brief Define a tab bar of one or more rows.
 *
 * In Resolve each row is a button row; the first starts on its first tab, the
 * rest with nothing selected. Anywhere else the whole bar becomes ONE Choice,
 * named after the first row, whose options are every row's tabs in order -- so
 * its value is the section index directly.
 *
 * Rows are never added to @p page: Resolve draws no pages and lays out in
 * definition order. Only the fallback Choice goes on the page.
 */
struct TabRowSpec { const char* name; const std::vector<TabSpec>* tabs; };

inline void DefineTabBar(OFX::ImageEffectDescriptor& desc, OFX::PageParamDescriptor* page,
                         const std::vector<TabRowSpec>& rows, const char* hint)
{
    if (rows.empty()) return;

    // All rows or none: a bar that came out half buttons, half Choice would
    // have two controls fighting over one selection.
    bool buttons = HostHasButtonRow() && !TabFacesDir(desc).empty();
    if (buttons)
    {
        for (std::size_t r = 0; r < rows.size(); ++r)
            if (!DefineButtonRow(desc, rows[r].name, hint, *rows[r].tabs, r == 0 ? 0 : -1))
                return;   // refused mid-way: nothing sensible left to do but leave it
        return;
    }

    OFX::ChoiceParamDescriptor* c = desc.defineChoiceParam(rows[0].name);
    c->setLabels("Section", "Section", "Section");
    c->setHint(hint);
    for (const TabRowSpec& r : rows)
        for (const TabSpec& t : *r.tabs) c->appendOption(t.label);
    c->setDefault(0);
    c->setAnimates(false);
    c->setEvaluateOnChange(false);
    c->setCanUndo(false);
    c->setIsPersistant(false);
    if (page) page->addChild(*c);
}

/** @brief Raw handle of a parameter, whatever its type; null if absent. */
inline OfxParamHandle RawParam(const OFX::ImageEffect* effect, const char* name,
                               OfxPropertySetHandle* outProps = nullptr)
{
    if (!effect) return nullptr;
    OfxParamSetHandle ps = nullptr;
    if (OFX::Private::gEffectSuite->getParamSet(effect->getHandle(), &ps) != kOfxStatOK || !ps)
        return nullptr;
    OfxParamHandle h = nullptr;
    if (OFX::Private::gParamSuite->paramGetHandle(ps, name, &h, outProps) != kOfxStatOK) return nullptr;
    return h;
}

/** @brief Whether this instance's parameter really is a button row (it is a
 *  Choice in every host but Resolve). */
inline bool IsButtonRow(OfxPropertySetHandle props)
{
    char* type = nullptr;
    return props
        && OFX::Private::gPropSuite->propGetString(props, kOfxParamPropType, 0, &type) == kOfxStatOK
        && type && std::strcmp(type, kOfxParamTypeButtonRow) == 0;
}

/** @brief Whether the named tab row exists and is a button row. */
inline bool TabRowIsButtons(const OFX::ImageEffect* effect, const char* name)
{
    OfxPropertySetHandle props = nullptr;
    return RawParam(effect, name, &props) && IsButtonRow(props);
}

/** @brief Selected tab of a row, as an index into @p tabs; -1 when nothing is
 *  selected, the row does not exist, or it cannot be read. For the fallback
 *  Choice this is simply its value (pass the full list of tabs). */
inline int GetTabRow(const OFX::ImageEffect* effect, const char* name,
                     const std::vector<TabSpec>& tabs)
{
    OfxPropertySetHandle props = nullptr;
    OfxParamHandle h = RawParam(effect, name, &props);
    if (!h) return -1;

    if (IsButtonRow(props))
    {
        const char* id = nullptr;
        if (OFX::Private::gParamSuite->paramGetValue(h, &id) != kOfxStatOK || !id) return -1;
        for (int i = 0; i < static_cast<int>(tabs.size()); ++i)
            if (std::strcmp(id, tabs[i].id) == 0) return i;
        return -1;
    }

    int v = -1;
    if (OFX::Private::gParamSuite->paramGetValue(h, &v) != kOfxStatOK) return -1;
    return (v >= 0 && v < static_cast<int>(tabs.size())) ? v : -1;
}

/** @brief Deselect every button of a button row (no effect on a Choice). Used
 *  so that picking a tab on one row clears the other. */
inline void ClearTabRow(const OFX::ImageEffect* effect, const char* name)
{
    OfxPropertySetHandle props = nullptr;
    OfxParamHandle h = RawParam(effect, name, &props);
    if (h && IsButtonRow(props)) OFX::Private::gParamSuite->paramSetValue(h, "");
}

}  // namespace rta
