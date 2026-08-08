// Draws each group as its own sprite at its own transform.
//
// The original static text is never drawn -- the output is composed purely of
// the animated copies, which is what makes the reveal read as the text itself
// moving rather than a ghost sliding over a solid title.
#pragma once

#include <vector>

#include "animator.h"
#include "image_view.h"
#include "segmentation.h"

namespace rta {

// dst is cleared to transparent, then every visible group is composited.
//
// `transforms` holds `taps` entries per group, laid out group-major:
// transforms[gi * taps + k]. With taps == 1 this is a plain sharp render; with
// more, the taps are averaged, which is what produces motion blur and defocus.
void compositeGroups(const ImageView& dst, const ImageView& src, const Segmentation& seg,
                     const std::vector<GroupTransform>& transforms, int taps,
                     const RectI& renderWindow);

}  // namespace rta
