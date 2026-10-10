"""Draw the VR Photo Studio logo (a VR headset whose lenses show a little photo) with Pillow.

Writes a multi-resolution Windows icon (16-256 px) used by the executables and the
installer, and optionally PNGs (e.g. for the GUI window icon)::

    python packaging/make_icon.py                      # -> packaging/icon.ico
    python packaging/make_icon.py --out x.ico --png logo.png --png-size 512

Every size is rendered separately at 4x and downsampled, and the smallest sizes use a
simplified drawing so the icon stays crisp in the taskbar and Explorer.
Deterministic: the same Pillow version always produces the same bytes.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

HERE = Path(__file__).resolve().parent
DEFAULT_ICO = HERE / "icon.ico"
ICO_SIZES: tuple[int, ...] = (16, 20, 24, 32, 40, 48, 64, 96, 128, 256)

# palette
BG_TOP = (56, 132, 255)  # bright blue
BG_BOTTOM = (124, 58, 237)  # violet
HEADSET = (250, 250, 252)
HEADSET_SHADE = (214, 218, 232)
LENS = (24, 22, 64)  # deep navy
SKY = (64, 150, 255)
SUN = (255, 196, 46)
HILL_BACK = (46, 196, 140)
HILL_FRONT = (22, 150, 110)
GLINT = (255, 255, 255, 90)


def _gradient(size: int, top: tuple[int, int, int], bottom: tuple[int, int, int]) -> Image.Image:
    """Vertical linear gradient, RGBA."""
    mask = Image.linear_gradient("L").resize((size, size), Image.Resampling.BILINEAR)
    a = Image.new("RGBA", (size, size), top + (255,))
    b = Image.new("RGBA", (size, size), bottom + (255,))
    return Image.composite(b, a, mask)


def _lens_picture(w: int, h: int, detail: bool) -> Image.Image:
    """The tiny 'photo' seen through one lens: sky, sun and two hills."""
    img = Image.new("RGBA", (w, h), LENS + (255,))
    if not detail:
        # small sizes: a single horizon so the lenses still read as "a picture"
        d = ImageDraw.Draw(img)
        d.rectangle((0, int(h * 0.62), w, h), fill=HILL_FRONT + (255,))
        return img
    sky = _gradient(max(w, h), SKY, LENS).resize((w, h))
    img.alpha_composite(sky)
    d = ImageDraw.Draw(img)
    r = 0.16 * h
    cx, cy = 0.70 * w, 0.32 * h
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=SUN + (255,))
    d.polygon([(-0.1 * w, h), (0.38 * w, 0.40 * h), (0.85 * w, h)], fill=HILL_BACK + (255,))
    d.polygon([(0.30 * w, h), (0.72 * w, 0.55 * h), (1.15 * w, h)], fill=HILL_FRONT + (255,))
    return img


def draw_logo(size: int, *, supersample: int = 4) -> Image.Image:
    """Render the logo at ``size`` x ``size`` (RGBA)."""
    if size < 8:
        raise ValueError("icon size must be at least 8 px")
    S = size * supersample
    detail = size >= 40
    canvas = Image.new("RGBA", (S, S), (0, 0, 0, 0))

    # rounded-square background with gradient
    bg_mask = Image.new("L", (S, S), 0)
    margin = 0.02 * S if size >= 32 else 0.0
    ImageDraw.Draw(bg_mask).rounded_rectangle(
        (margin, margin, S - 1 - margin, S - 1 - margin), radius=0.22 * S, fill=255
    )
    canvas.paste(_gradient(S, BG_TOP, BG_BOTTOM), (0, 0), bg_mask)

    # headset body (white) with nose cut-out, plus side strap stubs
    body = Image.new("L", (S, S), 0)
    bd = ImageDraw.Draw(body)
    x0, y0, x1, y1 = 0.12 * S, 0.29 * S, 0.88 * S, 0.71 * S
    bd.rounded_rectangle((x0, y0, x1, y1), radius=0.13 * S, fill=255)
    strap_h = 0.13 * S
    bd.rectangle((0.03 * S, 0.5 * S - strap_h / 2, x0 + 0.05 * S, 0.5 * S + strap_h / 2), fill=255)
    bd.rectangle((x1 - 0.05 * S, 0.5 * S - strap_h / 2, 0.97 * S, 0.5 * S + strap_h / 2), fill=255)
    nose_r = 0.095 * S
    bd.ellipse((0.5 * S - nose_r, y1 - nose_r * 0.9, 0.5 * S + nose_r, y1 + nose_r * 1.1), fill=0)
    # keep everything inside the rounded background
    body = ImageChops.darker(body, bg_mask)
    shade = Image.new("RGBA", (S, S), HEADSET_SHADE + (255,))
    face = Image.new("RGBA", (S, S), HEADSET + (255,))
    # subtle bottom shading on the body for depth
    shade_mask = Image.linear_gradient("L").resize((S, S)).point(lambda v: 255 if v > 175 else 0)
    face = Image.composite(shade, face, shade_mask)
    canvas.paste(face, (0, 0), body)

    # lenses
    lens_w, lens_h = 0.25 * S, 0.24 * S
    lens_y = 0.37 * S
    for lx in (0.205 * S, 0.545 * S):
        box = (int(lx), int(lens_y), int(lx + lens_w), int(lens_y + lens_h))
        w, h = box[2] - box[0], box[3] - box[1]
        pic = _lens_picture(w, h, detail)
        lens_mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(lens_mask).rounded_rectangle((0, 0, w - 1, h - 1), radius=0.32 * min(w, h), fill=255)
        canvas.paste(pic, box[:2], lens_mask)
        if detail:  # glossy glint across the top-left of each lens
            glint = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            ImageDraw.Draw(glint).ellipse((-0.4 * w, -0.9 * h, 0.75 * w, 0.42 * h), fill=GLINT)
            clipped = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            clipped.paste(glint, (0, 0), lens_mask)
            canvas.alpha_composite(clipped, box[:2])

    return canvas.resize((size, size), Image.Resampling.LANCZOS)


def write_ico(path: Path = DEFAULT_ICO, sizes: Sequence[int] = ICO_SIZES) -> Path:
    """Write a multi-size .ico (BMP frames, which every Windows tool accepts)."""
    sizes = sorted({int(s) for s in sizes if 8 <= int(s) <= 256})
    if not sizes:
        raise ValueError("no valid icon sizes (8..256)")
    frames = [draw_logo(s) for s in sizes]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    largest = frames[-1]
    largest.save(
        path,
        format="ICO",
        sizes=[(s, s) for s in sizes],
        append_images=frames[:-1],
        bitmap_format="bmp",
    )
    return path


def write_png(path: Path, size: int = 256) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    draw_logo(size).save(path, format="PNG", optimize=True)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Draw the VR Photo Studio icon.")
    ap.add_argument("--out", type=Path, default=DEFAULT_ICO, help=f"icon file (default {DEFAULT_ICO})")
    ap.add_argument("--sizes", type=int, nargs="+", default=list(ICO_SIZES), help="icon sizes in px")
    ap.add_argument("--png", type=Path, default=None, help="also write a PNG here")
    ap.add_argument("--png-size", type=int, default=256, help="PNG size in px (default 256)")
    args = ap.parse_args(argv)
    ico = write_ico(args.out, args.sizes)
    sys.stdout.write(f"wrote {ico}\n")
    if args.png:
        sys.stdout.write(f"wrote {write_png(args.png, args.png_size)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
