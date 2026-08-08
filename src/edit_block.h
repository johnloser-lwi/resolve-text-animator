// Reference-counted wrapper around OFX's parameter edit block.
//
// The OFX spec is explicit that when a plugin calls paramSetValue itself --
// "either from custom GUI interaction or some analysis of imagery" -- the
// writes must be bracketed by paramEditBegin/paramEditEnd. Every parameter this
// plugin writes is exactly that case: the curve editor's handles.
//
// Without the brackets the host has been told nothing about the edit. Resolve
// tolerates it; Fusion does not, and keeps serving cached frames after an
// overlay drag while edits made through its own Inspector -- which the host
// performs itself, and therefore already knows about -- update correctly. That
// asymmetry is the signature of this bug.
//
// Bracketing also collapses a drag into a single undo step instead of one per
// mouse-move event, which is what the block was designed for.
#pragma once

#include <map>

#include "ofxsImageEffect.h"

namespace rta {
namespace detail {

// Open-block depth per effect instance. Thread-local because these calls only
// ever happen on the host's UI thread, so a shared map would need a lock for no
// benefit.
inline int& editDepth(OFX::ImageEffect* effect) {
  static thread_local std::map<OFX::ImageEffect*, int> depths;
  return depths[effect];
}

}  // namespace detail

// Open a block, or note that one is already open.
//
// Counted, and that is the point: a paramSetValue inside a block is dispatched
// by the host straight back into changedParam, where a handler may open another.
// OFX does not define what nesting means, and a host that treats begin/end as a
// flag rather than a counter breaks on it -- the innermost end closes the outer
// block and the ends that follow are unmatched, leaving parameter and interact
// state inconsistent for the rest of the session. Counting here means the host
// only ever sees one begin and one matching end.
inline void beginEdit(OFX::ImageEffect* effect, const char* name) {
  if (!effect) return;
  int& depth = detail::editDepth(effect);
  if (depth == 0) {
    // Raised only after the host accepts the block, so a throw cannot leave a
    // phantom level to unwind.
    try {
      effect->beginEditBlock(name);
    } catch (...) {
      return;
    }
  }
  ++depth;
}

// Close the outermost block; inner levels merely decrement.
inline void endEdit(OFX::ImageEffect* effect) {
  if (!effect) return;
  int& depth = detail::editDepth(effect);
  if (depth == 0) return;  // unmatched end: ignore rather than corrupt
  if (--depth == 0) {
    try {
      effect->endEditBlock();
    } catch (...) {
    }
  }
}

// Force every open level shut. A last resort for a drag that begins and never
// ends because the host never delivered the mouse-up -- released outside the
// viewer, a page switched mid-drag, a modal dialog taking the release. The
// block would otherwise stay open for the rest of the session.
inline void forceCloseEdits(OFX::ImageEffect* effect) {
  if (!effect) return;
  int& depth = detail::editDepth(effect);
  if (depth == 0) return;
  depth = 0;
  try {
    effect->endEditBlock();
  } catch (...) {
  }
}

}  // namespace rta
