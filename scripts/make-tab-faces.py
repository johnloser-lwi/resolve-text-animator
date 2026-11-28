#!/usr/bin/env python3
"""Generate the tab-button face images for Resolve's Inspector tab row.

Resolve draws a button-row button from an image per state (see src/button_row.h).
The label is set separately, as text, so these are plain faces -- a rounded
rectangle with an outline, a fill and, when selected, an accent underline -- one
PNG per state, sized to the button at 3x for Retina. Anti-aliased by
supersampling. Standard library only.

    python3 scripts/make-tab-faces.py      # writes packaging/tab-faces/*.png

Re-run after changing a size or colour here or in src/button_row.h; the PNGs are
committed so the build needs no Python.
"""

import os
import struct
import sys
import zlib

# Must match kButtonRowWidthPx / kButtonRowHeightPx in src/button_row.h.
WIDTH, HEIGHT, SCALE = 56, 22, 3

RADIUS_PT  = 3.0    # corner radius
BORDER_PT  = 1.0    # outline thickness
SUPERSAMPLE = 4     # per axis, for smooth corners

ACCENT = (0xF2, 0x8C, 0x28, 0xFF)          # warm orange: the selected tab's underline
CLEAR  = (0, 0, 0, 0)

# Face name -> (fill, outline, underline colour, underline thickness in points).
# Modelled on the tab faces of a shipping Resolve plugin, plus an outline so an
# unselected tab still reads as a button: the outline brightens on hover, and
# the selected tab adds a darker fill and a coloured bar along the bottom.
FACES = {
    "normal":   (CLEAR,                    (0x4A, 0x4A, 0x4A, 0xFF), CLEAR,  0),
    "hover":    ((0x2C, 0x2C, 0x2C, 0xFF), (0x5E, 0x5E, 0x5E, 0xFF), CLEAR,  0),
    "checked":  ((0x33, 0x33, 0x33, 0xFF), (0x4A, 0x4A, 0x4A, 0xFF), ACCENT, 2),
    "disabled": (CLEAR,                    (0x38, 0x38, 0x38, 0xFF), CLEAR,  0),
}


def png(width, height, rows):
    """An RGBA PNG from a list of `height` rows, each a list of RGBA tuples."""
    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + b"".join(bytes(px) for px in row) for row in rows)   # filter 0 per row
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)                    # 8-bit RGBA
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def inside_rounded(x, y, w, h, r):
    """Distance from (x, y) to the rounded rectangle's edge; > 0 means inside."""
    # Distance to the nearest edge for the straight parts...
    d = min(x, y, w - x, h - y)
    # ...and to the corner arc where both coordinates are within the radius.
    cx = r if x < r else (w - r if x > w - r else None)
    cy = r if y < r else (h - r if y > h - r else None)
    if cx is not None and cy is not None:
        d = r - ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
    return d


def face(fill, outline, line, line_pt):
    """One face at SCALE: a rounded rectangle with an outline and an optional
    underline, anti-aliased by averaging SUPERSAMPLE^2 samples per pixel."""
    w, h = WIDTH * SCALE, HEIGHT * SCALE
    r, bw, lw = RADIUS_PT * SCALE, BORDER_PT * SCALE, line_pt * SCALE
    n = SUPERSAMPLE
    rows = []
    for py in range(h):
        row = []
        for px in range(w):
            acc = [0.0, 0.0, 0.0, 0.0]          # premultiplied RGBA
            for sy in range(n):
                for sx in range(n):
                    x = px + (sx + 0.5) / n
                    y = py + (sy + 0.5) / n
                    d = inside_rounded(x, y, w, h, r)
                    if d <= 0:
                        continue
                    if lw and y >= h - lw:
                        c = line                 # the underline replaces the bottom edge
                    elif d <= bw:
                        c = outline
                    else:
                        c = fill
                    a = c[3] / 255.0
                    acc[0] += c[0] * a; acc[1] += c[1] * a; acc[2] += c[2] * a; acc[3] += a
            k = n * n
            alpha = acc[3] / k
            if alpha <= 0:
                row.append(CLEAR)
            else:
                row.append((round(acc[0] / acc[3]), round(acc[1] / acc[3]),
                            round(acc[2] / acc[3]), round(alpha * 255)))
        rows.append(row)
    return rows


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(repo, "packaging", "tab-faces")
    os.makedirs(out, exist_ok=True)
    for name, (fill, outline, line, line_pt) in FACES.items():
        path = os.path.join(out, name + ".png")
        with open(path, "wb") as f:
            f.write(png(WIDTH * SCALE, HEIGHT * SCALE, face(fill, outline, line, line_pt)))
        print(path)


if __name__ == "__main__":
    sys.exit(main())
