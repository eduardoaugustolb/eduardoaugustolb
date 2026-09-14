#!/usr/bin/env python3
"""Turn a photo into ascii.svg — a self-typing, dual-theme dot-matrix portrait.

Renders two layers in the same SVG: light draws the shadow (on a light
background) and dark draws the light (on a dark background, chalk-on-blackboard
style) — only one shows at a time, via display + prefers-color-scheme. That's
why the dark layer doesn't look like an inverted filter: it's its own mapping,
harmonious on each background.

"Jitter" style: stippled like the reference wallpaper — a short dot ramp
(" ", "·", "•", "●") + Floyd-Steinberg dithering + pseudo-random horizontal
offset per row. With only 4 levels, detail (glasses, teeth, contours) comes
from error diffusion and the higher resolution (COLS 120), not from the ramp.

    uv pip install --system pillow numpy opencv-python-headless rembg onnxruntime
    python3 scripts/make_portrait.py photo.png --crop 60,10,420,360
    python3 scripts/embed_portrait_font.py      # inline the font, see below

The first run downloads a ~176 MB background-removal model, once.
Pass --no-bg to skip it entirely (uses the whole photo, never cuts the head).

Two things decide whether the output is any good, and neither is a parameter:

  * The photo. Dots draw with shadow, not detail — 4 dot sizes in total. You
    need side light (a window at ~45°, everything else off), a tight crop from
    chin to just above the hair, and real resolution. A 320px headshot fails:
    thin features like glasses frames are averaged away on downscale. Flat
    frontal light renders the face as a hole.
  * The darkening curve below. Without it the face comes out washed out and
    featureless — brows, glasses and lips all dissolve.

The grid bakes in an advance width of exactly 0.600 em (CHAR_W / FONT_SIZE), so
after generating, run scripts/embed_portrait_font.py to inline JetBrains Mono.
Otherwise a viewer whose default monospace is narrower — Consolas is ≈0.55 —
sees the portrait about 7% too narrow.

Motion is SMIL, because GitHub strips <script> from READMEs: each row is
revealed by a clipPath wipe with a cursor block riding its edge, staggered top
to bottom, frozen at the end so it prints once and stops.
"""
import argparse
import sys

import cv2
import numpy as np
from PIL import Image
from rembg import new_session, remove

# u2net (~176 MB, cached at ~/.u2net in CI) instead of the default bria-rmbg-2.0
# (~1 GB): faster and good enough to separate the bust from the background.
_BG_SESSION = None


def _bg_session():
    global _BG_SESSION
    if _BG_SESSION is None:
        _BG_SESSION = new_session("u2net")
    return _BG_SESSION

RAMP = " \u00b7\u2022\u25cf"  # jitter/dot halftone: blank, middle dot, bullet, black circle
# bright/sparse -> dark/dense. Few levels on purpose (like the stippled
# wallpaper): detail comes from error-diffusion dithering + higher resolution,
# not from a long ramp. Dots only breaks vertical "banding" and mimics the
# reference's jittered stipple.
COLS = 120                 # more columns = more detail with a short ramp
CLAHE_CLIP = 2.0           # lower than before: with only 4 levels, texture turns into noise
GAMMA = 1.0                # ramp mapping exponent
CURVE = 1.35               # light layer: darkens to hold the midtones
DARK_CURVE = 1.0           # dark layer: no boost, brightness already shapes it on its own
CROP_BOTTOM = 0.0          # fraction to trim off the bottom (torso, chair)
ROW_RATIO = 0.50           # monospace cells are about twice as tall as wide
DITHER = True              # Floyd-Steinberg: preserves glasses, teeth, contours
JITTER = 0.45              # horizontal offset per row (fraction of CHAR_W)
JITTER_SEED = 7            # fixed so the portrait is deterministic across runs
VIGNETTE = True            # gentle vertical fade: the top (hair)
VIG_TOP = 0.10             # and bottom (chin) bands taper instead of cutting
VIG_BOT = 0.06             # straight off — sides and face size untouched

FG_LIGHT = "#6e7681"       # readable on GitHub light — the portrait's grey
FG_DARK = "#c9d1d9"        # and its dark-mode step
CHAR_W = 7.74              # 0.600 em at FONT_SIZE — keep these in step
FONT_SIZE = 12.9
LINE_H = 15
ROW_DELAY = 0.09           # per-row stagger, seconds
FAMILY = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"


def prep(path, crop=None, no_bg=False, invert=False, curve=None):
    """Cut out the background, even out the local contrast, then darken.

    rembg sometimes bites into the head (dark hair on a dark background reads
    as "background"). To never cut into the subject:

      * a wide closing fills the hair bays/notches;
      * local notch-fill: only columns sunk relative to their close
        neighborhood (±10% of the width) become subject again, up to 12% of
        the height and only in the central head region — without pulling in
        dark background that would turn into a solid bar at the top;
      * short dilation + minimal feather, no gray halo on the background;
      * if the mask comes back nearly empty (<15%), discard it and use the
        whole photo;
      * --no-bg skips removal and uses the whole photo (recommended if the
        background is light or the hair keeps getting bitten).

    invert=True renders the dark layer: the subject is composited over black
    and the scale is inverted, so the dots draw the LIGHT (chalk on blackboard)
    instead of the shadow — harmonious on the dark background instead of a
    white block.
    """
    src = Image.open(path).convert("RGBA")
    if crop:
        src = src.crop(crop)

    if no_bg:
        gray = np.array(src.convert("L"))
    else:
        try:
            cut = remove(src, session=_bg_session())
            alpha = np.array(cut.split()[-1])
        except Exception:
            cut, alpha = None, None

        coverage = 0.0
        if alpha is not None:
            coverage = float((alpha > 20).mean())

        if alpha is None or coverage < 0.15:
            # Nearly empty mask (= ate the head): don't risk it, use everything.
            # High coverage (>97%) is normal on a tight crop — it just means
            # the subject fills the frame; the mask is still valid.
            gray = np.array(src.convert("L"))
        else:
            # The mask usually bites into the hair (dark on dark background):
            # close bays/notches with a wide closing, ...
            w = max(src.size)
            kc = max(9, int(w * 0.07) | 1)
            binm = (alpha > 20).astype("uint8") * 255
            closed = cv2.morphologyEx(
                binm, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)))
            # ...local notch-fill: each column's reference top is the median
            # of tops in its neighborhood (±10% of the width). Only fills when
            # the column is sunk >3px and <=12% of the height, and only in the
            # central 60% (head) — dark background at the edges never becomes
            # subject, so no solid bar forms.
            fg_px = closed > 127
            h, ww = fg_px.shape
            has = fg_px.any(axis=0)
            tops = np.where(has, fg_px.argmax(axis=0), h)
            win = max(5, ww // 10)
            idx = np.arange(ww)
            lo = np.clip(idx - win, 0, ww - 1)
            hi = np.clip(idx + win + 1, 0, ww)
            local = np.empty(ww, dtype=float)
            for x in range(ww):
                seg = tops[lo[x]:hi[x]]
                seg = seg[seg < h]
                local[x] = np.median(seg) if seg.size else h
            depth = tops - local
            center = (idx >= ww * 0.2) & (idx <= ww * 0.8)
            fix = (has & center & (tops < h * 0.5)
                   & (depth > 3) & (depth <= h * 0.12))
            if fix.any():
                rows = np.arange(h)[:, None]
                fill = (fix[None, :]
                        & (rows >= local[None, :].astype(int))
                        & (rows < tops[None, :]))
                closed[fill] = 255
            # ...recover edge strands with a short dilation, ...
            kd = max(5, int(w * 0.02) | 1)
            dil = cv2.dilate(closed, np.ones((kd, kd), np.uint8))
            # ...and feather just enough to avoid a gray halo on the background.
            soft = cv2.GaussianBlur(dil.astype("float32"), (0, 0), sigmaX=2.0)
            soft = (soft / 255.0)[..., None].astype("float32")
            fg = np.array(src.convert("RGB"), dtype="float32")
            # light layer composites over white (shadow -> dot);
            # dark layer composites over black (light -> dot).
            paper = 0.0 if invert else 255.0
            comp = fg * soft + paper * (1.0 - soft)
            gray = comp[..., 0] * 0.299 + comp[..., 1] * 0.587 + comp[..., 2] * 0.114
            gray = np.clip(gray, 0, 255).astype("uint8")

    if curve is None:
        curve = DARK_CURVE if invert else CURVE
    gray = cv2.bilateralFilter(gray, 7, 35, 35)       # light touch: holds glasses/teeth
    gray = cv2.createCLAHE(clipLimit=CLAHE_CLIP,
                           tileGridSize=(8, 8)).apply(gray)
    gray = (255.0 * (gray / 255.0) ** curve).astype("uint8")
    if invert:
        gray = 255 - gray  # brightness becomes density: background/hair -> blank
        # Outside the mask the inverted image is never pure: background texture
        # + feather turn into dots via dithering (starfield). Zero everything
        # mostly outside the subject; the feathered edge still gives the soft
        # contour.
        try:
            outside = soft[..., 0] < 0.35
            gray[outside] = 255
        except NameError:
            gray[gray > 245] = 255  # no mask (--no-bg): only the near-white
    if VIGNETTE:
        # By now blank == 255 on both layers (light composites over white;
        # dark already inverted and zeroed the background). The fade pushes
        # the top and bottom bands toward blank with a smoothstep, so hair /
        # chin touching the crop taper instead of cutting straight off.
        # Vertical only: sides, ears and width untouched.
        h, w = gray.shape
        yy = np.arange(h, dtype="float32")[:, None] / h
        f_top = np.clip((VIG_TOP - yy) / VIG_TOP, 0, 1)
        f_bot = np.clip((yy - (1 - VIG_BOT)) / VIG_BOT, 0, 1)
        f = np.maximum(f_top, f_bot)
        f = f * f * (3 - 2 * f)  # smoothstep: transition without a hard step
        f = np.broadcast_to(f, (h, w))
        gray = (gray * (1 - f) + 255.0 * f).astype("uint8")
    return Image.fromarray(gray)


def to_lines(img, cols=COLS, gamma=GAMMA, dither=DITHER):
    """Quantize to the dot ramp with error diffusion (jitter).

    With only 4 levels, direct quantization posterizes and erases glasses /
    teeth. Floyd-Steinberg pushes the error to the neighbors, preserving
    texture and fine contours as stipple — the same principle as the wallpaper.
    """
    w, h = img.size
    if CROP_BOTTOM:
        img = img.crop((0, 0, w, int(h * (1 - CROP_BOTTOM))))
        w, h = img.size

    rows = int(cols * (h / w) * ROW_RATIO)
    img = img.resize((cols, rows), Image.LANCZOS)
    buf = np.array(img, dtype="float32") / 255.0   # 1 = branco, 0 = preto
    n = len(RAMP)

    # "darkness" levels 0..1 evenly spaced; gamma applies on the way in
    def quant(v):
        v = min(1.0, max(0.0, v))
        dark = (1.0 - v) ** gamma
        idx = min(n - 1, int(dark * n))
        # representative level value (bucket center) for computing the error
        rep = 1.0 - (idx + 0.5) / n
        if idx == 0:
            rep = 1.0
        elif idx == n - 1:
            rep = 1.0 - (n - 0.5) / n
        # approximately undo the gamma
        if gamma != 1.0:
            rep = 1.0 - (1.0 - rep) ** (1.0 / gamma) if rep < 1.0 else 1.0
        return idx, rep

    idx_map = np.zeros((rows, cols), dtype=np.int32)
    if dither:
        for y in range(rows):
            for x in range(cols):
                idx, rep = quant(buf[y, x])
                idx_map[y, x] = idx
                err = buf[y, x] - rep
                if x + 1 < cols:
                    buf[y, x + 1] += err * 7 / 16
                if y + 1 < rows:
                    if x > 0:
                        buf[y + 1, x - 1] += err * 3 / 16
                    buf[y + 1, x] += err * 5 / 16
                    if x + 1 < cols:
                        buf[y + 1, x + 1] += err * 1 / 16
    else:
        for y in range(rows):
            for x in range(cols):
                idx_map[y, x] = quant(buf[y, x])[0]

    out = []
    for r in range(rows):
        out.append("".join(RAMP[i] for i in idx_map[r]).rstrip())

    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    return out


def build_svg(lines_light, lines_dark=None, cols=COLS):
    """Assemble the SVG with the two themed layers.

    lines_light draws the shadow (light background); lines_dark draws the
    light (dark background). Only one layer is visible at a time, via display
    + media query — the same file serves both GitHub schemes.
    """
    layers = [("l", "ll", lines_light)]
    if lines_dark is not None:
        layers.append(("d", "ld", lines_dark))

    pad = 14
    width = int(cols * CHAR_W + (pad + JITTER * CHAR_W + 1) + pad)
    height = max(len(lines) for _, _, lines in layers) * LINE_H + pad * 2

    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
         f'height="{height}" viewBox="0 0 {width} {height}" '
         f'font-family="{FAMILY}">',
         f'<style>.a{{fill:{FG_LIGHT}}}.ld{{display:none}}'
         f'@media(prefers-color-scheme:dark){{.a{{fill:{FG_DARK}}}'
         f'.ll{{display:none}}.ld{{display:inline}}}}</style>']

    import random
    rng = random.Random(JITTER_SEED)
    # per-row jitter: shifts each row by ±JITTER*CHAR_W to break the vertical
    # alignment — the wallpaper's "shaky" stipple. The extra pad absorbs the
    # rightward shift without clipping. Same seed on both layers, so the
    # geometry matches.
    nrows = max(len(lines) for _, _, lines in layers)
    shifts = [(rng.random() * 2 - 1) * JITTER * CHAR_W for _ in range(nrows)]
    pad_l = pad + JITTER * CHAR_W + 1

    for prefix, cls, lines in layers:
        p.append(f'<g class="{cls}">')
        for i, line in enumerate(lines):
            y = pad + i * LINE_H
            x = pad_l + shifts[i]
            begin = f"{i * ROW_DELAY:.2f}s"
            end = f"{(i + 1) * ROW_DELAY:.2f}s"
            w = max(len(line), 1) * CHAR_W
            safe = (line.replace("&", "&amp;").replace("<", "&lt;")
                        .replace(">", "&gt;"))

            p.append(f'<clipPath id="{prefix}{i}"><rect x="{x:.1f}" y="{y}" '
                     f'height="{LINE_H}" width="0">'
                     f'<animate attributeName="width" from="0" to="{w:.1f}" '
                     f'begin="{begin}" dur="{ROW_DELAY}s" fill="freeze"/>'
                     f'</rect></clipPath>')
            p.append(f'<g clip-path="url(#{prefix}{i})">'
                     f'<text xml:space="preserve" '
                     f'x="{x:.1f}" y="{y + 11.2:.1f}" class="a" '
                     f'font-size="{FONT_SIZE}">{safe}</text></g>')
            # the cursor: a small block riding the wipe edge, gone once the row lands
            p.append(f'<rect y="{y + 1}" width="6" height="12" class="a" '
                     f'opacity="0">'
                     f'<animate attributeName="x" from="{x:.1f}" to="{x + w:.1f}" '
                     f'begin="{begin}" dur="{ROW_DELAY}s" fill="freeze"/>'
                     f'<set attributeName="opacity" to="0.8" begin="{begin}"/>'
                     f'<set attributeName="opacity" to="0" begin="{end}"/></rect>')
        p.append("</g>")

    p.append("</svg>")
    return "".join(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("photo")
    ap.add_argument("out", nargs="?", default="ascii.svg")
    ap.add_argument("--crop", help="left,top,right,bottom, applied first — crop "
                                   "tight to the head so the whole grid goes to "
                                   "the face")
    ap.add_argument("--cols", type=int, default=COLS)
    ap.add_argument("--no-bg", action="store_true",
                    help="skip rembg and use the whole photo — use if the "
                         "hair/head keeps getting cut")
    ap.add_argument("--no-dither", action="store_true",
                    help="turn off Floyd-Steinberg (posterizes, less detail)")
    ap.add_argument("--preview", action="store_true",
                    help="print the ASCII to the terminal as well")
    ap.add_argument("--out-txt", metavar="PATH",
                    help="also write the plain ASCII to this file, e.g. "
                         "~/.config/fastfetch/logo.txt")
    ap.add_argument("--out-txt-dark", metavar="PATH",
                    help="plain ASCII of the dark layer (light -> dot)")
    ap.add_argument("--no-dark", action="store_true",
                    help="render only the light layer (single-theme portrait)")
    args = ap.parse_args()

    crop = None
    if args.crop:
        parts = [int(v) for v in args.crop.split(",")]
        if len(parts) != 4:
            sys.exit("--crop needs four numbers: left,top,right,bottom")
        crop = tuple(parts)

    dither = not args.no_dither
    lines = to_lines(prep(args.photo, crop, no_bg=args.no_bg),
                     cols=args.cols, dither=dither)
    lines_dark = None
    if not args.no_dark:
        lines_dark = to_lines(
            prep(args.photo, crop, no_bg=args.no_bg, invert=True),
            cols=args.cols, dither=dither)
    if args.preview:
        print("\n".join(lines))
    if args.out_txt:
        with open(args.out_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"wrote {args.out_txt} — {len(lines)} rows, {args.cols} columns")
    if args.out_txt_dark and lines_dark is not None:
        with open(args.out_txt_dark, "w", encoding="utf-8") as f:
            f.write("\n".join(lines_dark) + "\n")
        print(f"wrote {args.out_txt_dark} — {len(lines_dark)} rows, "
              f"{args.cols} columns")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(build_svg(lines, lines_dark, cols=args.cols))
    dark_info = f" + {len(lines_dark)} dark" if lines_dark is not None else ""
    print(f"wrote {args.out} — {len(lines)} rows{dark_info}, {args.cols} columns")
    print("next: python3 scripts/embed_portrait_font.py")


if __name__ == "__main__":
    main()
