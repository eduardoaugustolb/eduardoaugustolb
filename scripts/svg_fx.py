#!/usr/bin/env python3
"""SMIL animation helpers for the profile README's SVG widgets.

Reimplements three React-Bits-style effects without JavaScript — GitHub
strips <script> from READMEs and these SVGs load through <img>, so GSAP /
motion could never run here. Everything below is precomputed frames plus
SMIL, the same technique as the portrait's typing reveal in ascii.svg:

  * shuffle loop  (Shuffle, loop=true): the clean text holds, then a fast
    scramble burst rolls across it, then it holds again, indefinitely;
  * decrypt entry (DecryptedText, animateOn="view"): scrambled frames settle
    into the clean text once on load, then stay put;
  * shine sweep   (ShinyText): a bright band travels across a gradient fill,
    in an indefinite loop, theme-aware via CSS stop colors.

Robustness pattern: the clean text is always rendered as a static base
element, and the scramble frames are overlays that cover it only during
their slot. A renderer without SMIL (or with animations disabled) simply
shows the clean text — nothing ever goes invisible.

Determinism: every scramble uses a seeded RNG keyed by the text itself, so
regenerating with unchanged data produces byte-identical output and the
scheduled workflow never commits noise.

Standard library only, like generate_stats.py.
"""
import random

# Every glyph here is in the inlined JBMono subsets (basic latin coverage was
# verified against jbmono-400/600), so frames never tofu on any viewer.
SCRAMBLE = "!<>-_\\/[]{}=+*^?#"


def esc(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def scramble_frames(text, n=6, seed=""):
    """Progressively-revealing scramble frames (never includes the clean text).

    Frame i keeps the first (i+1)*len/(n+1) characters and scrambles the rest,
    left to right like a decrypt settling. Seeded, so stable between runs.
    """
    rng = random.Random(f"fx:{seed}:{text}")
    out = []
    total = len(text)
    for i in range(n):
        keep = (i + 1) * total // (n + 1)
        buf = [ch if ch == " " or j < keep else rng.choice(SCRAMBLE)
               for j, ch in enumerate(text)]
        out.append("".join(buf))
    return out


def _attrs(x, y, size, fill=None, cls=None, anchor="start",
           weight=None, spacing=None):
    a = "" if anchor == "start" else f' text-anchor="{anchor}"'
    w = "" if weight is None else f' font-weight="{weight}"'
    ls = "" if spacing is None else f' letter-spacing="{spacing}"'
    f = f' fill="{fill}"' if fill else ""
    c = f' class="{cls}"' if cls else ""
    return f'x="{x}" y="{y}" font-size="{size}"{a}{w}{ls}{f}{c}'


def animated_text(x, y, text, size, fill=None, cls=None, anchor="start",
                  weight=None, spacing=None, mode="loop",
                  begin=0.20, slot=0.07, hold=3.40, frames=6, seed=""):
    """A text with a shuffle/decrypt animation, returned as an SVG fragment.

    mode="loop":  clean text holds `hold` seconds, then the scramble frames
                  flash past in `frames * slot` seconds, repeating forever.
    mode="entry": the scramble frames play once from `begin`, then the clean
                  base remains (DecryptedText-on-view equivalent).
    """
    text = str(text)
    attrs = _attrs(x, y, size, fill, cls, anchor, weight, spacing)
    parts = [f'<text xml:space="preserve" {attrs}>{esc(text)}</text>']
    inter = scramble_frames(text, frames, seed or text)
    n = len(inter)

    if mode == "entry":
        dur = n * slot
        kt = ";".join(f"{j / n:.4f}" for j in range(n + 1))
        for i, fr in enumerate(inter):
            vals = ["0"] * (n + 1)
            vals[i] = "1"
            parts.append(
                f'<text xml:space="preserve" {attrs} opacity="0">'
                f'<animate attributeName="opacity" values="{";".join(vals)}" '
                f'keyTimes="{kt}" dur="{dur:.2f}s" begin="{begin:.2f}s" '
                f'calcMode="discrete" fill="freeze"/>{esc(fr)}</text>')
    else:
        burst = n * slot
        dur = hold + burst
        bounds = [0.0, hold] + [hold + (j + 1) * slot for j in range(n)]
        kt = ";".join(f"{b / dur:.4f}" for b in bounds)
        for i, fr in enumerate(inter):
            vals = ["0"] * (n + 2)
            vals[i + 1] = "1"
            parts.append(
                f'<text xml:space="preserve" {attrs} opacity="0">'
                f'<animate attributeName="opacity" values="{";".join(vals)}" '
                f'keyTimes="{kt}" dur="{dur:.2f}s" begin="{begin:.2f}s" '
                f'calcMode="discrete" repeatCount="indefinite"/>{esc(fr)}</text>')
    return "".join(parts)


def shine_style():
    """CSS for the shine gradient stops, in both color schemes."""
    return (".sh0{stop-color:#6e7681}.sh1{stop-color:#ffffff}"
            "@media(prefers-color-scheme:dark)"
            "{.sh0{stop-color:#c9d1d9}.sh1{stop-color:#f0f6fc}}")


def shine_grad(gid, dur=3.60, begin=0.0):
    """A <linearGradient> with a bright band sweeping across, looping forever.

    Apply with fill="url(#gid)" on the element — no fill-setting class on the
    same element, since CSS fill would override the url() presentation attr.
    """
    return (f'<linearGradient id="{gid}" x1="0" y1="0" x2="1" y2="0">'
            f'<stop offset="0" class="sh0"/><stop offset="0.40" class="sh0"/>'
            f'<stop offset="0.50" class="sh1"/><stop offset="0.60" class="sh0"/>'
            f'<stop offset="1" class="sh0"/>'
            f'<animateTransform attributeName="gradientTransform" '
            f'type="translate" from="-1.1 0" to="1.1 0" dur="{dur:.2f}s" '
            f'begin="{begin:.2f}s" repeatCount="indefinite"/></linearGradient>')
