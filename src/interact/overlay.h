// Viewer overlay, wired up through the raw OFX interact entry point.
//
// The C++ Support library in upstream OpenFX ships ofxDrawSuite.h but never
// binds it: there is no gDrawSuite, and DrawArgs carries no draw context. The
// context only ever arrives in the draw action's inArgs, so the Support lib's
// interact wrapper cannot reach it at all.
//
// Rather than patch a vendored SDK that a fresh `git clone` would silently
// revert, this registers its own interact main entry and reads the raw action
// arguments directly.
#pragma once

#include "ofxsImageEffect.h"

namespace rta {

// Attaches the overlay to an effect descriptor. Safe to call when the host has
// no draw suite: the overlay then simply never draws.
void registerOverlay(OFX::ImageEffectDescriptor& desc);

}  // namespace rta
