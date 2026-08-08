// Draws each detected group's bounding box in a cycling colour.
// This is the tool that tells "the animation is wrong" apart from "the
// segmentation is wrong", which are otherwise indistinguishable on screen.
#pragma once

#include "image_view.h"
#include "segmentation.h"

namespace rta {

void drawDiagnostics(const ImageView& dst, const Segmentation& seg, int lineWidth = 2);

}  // namespace rta
