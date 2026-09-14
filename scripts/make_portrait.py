#!/usr/bin/env python3
"""Turn a photo into ascii.svg — a self-typing, dual-theme dot-matrix portrait.

Gera duas camadas no mesmo SVG: light desenha a sombra (fundo claro) e dark
desenha a luz (fundo escuro, estilo giz no quadro-negro) — so uma aparece por
vez, via display + prefers-color-scheme. E por isso que o dark nao parece um
filtro invertido: e um mapeamento proprio, harmonico em cada fundo.

Estilo "jitter": pontilhado como o wallpaper de referencia — rampa curta de
pontos (" ", "·", "•", "●") + dithering Floyd-Steinberg + deslocamento
horizontal pseudo-aleatorio por linha. Com so 4 niveis, o detalhe (oculos,
dentes, contornos) vem da difusao de erro e da resolucao maior (COLS 120),
nao da rampa.

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

# u2net (~176 MB, cacheado em ~/.u2net no CI) em vez do padrao bria-rmbg-2.0
# (~1 GB): mais rapido e suficiente para separar busto do fundo.
_BG_SESSION = None


def _bg_session():
    global _BG_SESSION
    if _BG_SESSION is None:
        _BG_SESSION = new_session("u2net")
    return _BG_SESSION

RAMP = " \u00b7\u2022\u25cf"  # jitter/dot halftone: blank, middle dot, bullet, black circle
# bright/sparse -> dark/dense. Poucos niveis de proposito (igual ao wallpaper
# pontilhado): o detalhe vem do dithering por difusao de erro + resolucao maior,
# nao de uma rampa longa. Manter so pontos quebra o "banding" vertical e imita
# o pontilhado com jitter da referencia.
COLS = 120                 # mais colunas = mais detalhe com rampa curta
CLAHE_CLIP = 2.0           # menor que antes: com so 4 niveis, textura vira ruido
GAMMA = 1.0                # ramp mapping exponent
CURVE = 1.35               # camada light: escurece p/ segurar meios-tons
DARK_CURVE = 1.0           # camada dark: sem reforco, o brilho ja modela sozinho
CROP_BOTTOM = 0.0          # fraction to trim off the bottom (torso, chair)
ROW_RATIO = 0.50           # monospace cells are about twice as tall as wide
DITHER = True              # Floyd-Steinberg: preserva oculos, dentes, contornos
JITTER = 0.45              # deslocamento horizontal por linha (fração de CHAR_W)
JITTER_SEED = 7            # fixo p/ o retrato ser deterministico entre runs

FG_LIGHT = "#6e7681"       # readable on GitHub light — the portrait's grey
FG_DARK = "#c9d1d9"        # and its dark-mode step
CHAR_W = 7.74              # 0.600 em at FONT_SIZE — keep these in step
FONT_SIZE = 12.9
LINE_H = 15
ROW_DELAY = 0.09           # per-row stagger, seconds
FAMILY = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"


def prep(path, crop=None, no_bg=False, invert=False, curve=None):
    """Cut out the background, even the local contrast, then darken.

    O rembg as vezes morde a cabeca (cabelo escuro sobre fundo escuro vira
    "fundo"). Para nunca cortar o sujeito:

      * closing largo preenche as baias/recortes do cabelo;
      * notch-fill local: so as colunas afundadas em relacao a vizinhanca
        proxima (±10% da largura) voltam a ser sujeito, ate 12% da altura
        e apenas na regiao central da cabeca — sem puxar fundo escuro vira
        barra solida no topo;
      * dilatacao curta + feather minimo, sem halo cinza no fundo;
      * se a mascara vem quase vazia (<15%), descarta e usa a foto inteira;
      * --no-bg pula a remocao e usa a foto inteira (recomendado se o fundo
        for claro ou se o cabelo continuar sendo mordido).

    invert=True gera a camada dark: o sujeito e composto sobre preto e a
    escala e invertida, entao os pontos desenham a LUZ (giz no quadro-negro)
    em vez da sombra — harmonico no fundo escuro em vez de um bloco branco.
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
            # Mascara quase vazia (= comeu a cabeca): nao arrisca, usa tudo.
            # Cobertura alta (>97%) e normal em crop apertado — so significa
            # que o sujeito preenche o quadro; a mascara continua valida.
            gray = np.array(src.convert("L"))
        else:
            # A mascara costuma morder o cabelo (escuro sobre fundo escuro):
            # fecha baias/recortes com closing largo, ...
            w = max(src.size)
            kc = max(9, int(w * 0.07) | 1)
            binm = (alpha > 20).astype("uint8") * 255
            closed = cv2.morphologyEx(
                binm, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)))
            # ...notch-fill local: o topo de referencia de cada coluna e a
            # mediana dos topos na vizinhanca (±10% da largura). So preenche
            # se a coluna esta afundada >3px e <=12% da altura, e apenas nos
            # 60% centrais (cabeca) — fundo escuro das bordas nunca vira
            # sujeito, entao nao forma barra solida.
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
            # ...recupera fios da borda com dilatacao curta, ...
            kd = max(5, int(w * 0.02) | 1)
            dil = cv2.dilate(closed, np.ones((kd, kd), np.uint8))
            # ...e featheriza so o minimo para nao deixar halo cinza no fundo.
            soft = cv2.GaussianBlur(dil.astype("float32"), (0, 0), sigmaX=2.0)
            soft = (soft / 255.0)[..., None].astype("float32")
            fg = np.array(src.convert("RGB"), dtype="float32")
            # camada light compoe sobre branco (sombra -> ponto);
            # camada dark compoe sobre preto (luz -> ponto).
            paper = 0.0 if invert else 255.0
            comp = fg * soft + paper * (1.0 - soft)
            gray = comp[..., 0] * 0.299 + comp[..., 1] * 0.587 + comp[..., 2] * 0.114
            gray = np.clip(gray, 0, 255).astype("uint8")

    if curve is None:
        curve = DARK_CURVE if invert else CURVE
    gray = cv2.bilateralFilter(gray, 7, 35, 35)       # leve: segura oculos/dentes
    gray = cv2.createCLAHE(clipLimit=CLAHE_CLIP,
                           tileGridSize=(8, 8)).apply(gray)
    gray = (255.0 * (gray / 255.0) ** curve).astype("uint8")
    if invert:
        gray = 255 - gray  # brilho vira densidade: fundo/hair -> blank
        # Fora da mascara o invertido nunca e puro: textura do fundo + o
        # feather viram pontinho via dithering (starfield). Zera tudo que
        # esta majoritariamente fora do sujeito; a borda featherizada
        # continua dando o contorno suave.
        try:
            outside = soft[..., 0] < 0.35
            gray[outside] = 255
        except NameError:
            gray[gray > 245] = 255  # sem mascara (--no-bg): so o quase-branco
    return Image.fromarray(gray)


def to_lines(img, cols=COLS, gamma=GAMMA, dither=DITHER):
    """Quantiza para a rampa de pontos com difusao de erro (jitter).

    Com so 4 niveis, a quantizacao direta posteriza e apaga oculos/dentes.
    O Floyd-Steinberg empurra o erro para os vizinhos, preservando textura e
    contornos finos como pontilhado — o mesmo principio do wallpaper.
    """
    w, h = img.size
    if CROP_BOTTOM:
        img = img.crop((0, 0, w, int(h * (1 - CROP_BOTTOM))))
        w, h = img.size

    rows = int(cols * (h / w) * ROW_RATIO)
    img = img.resize((cols, rows), Image.LANCZOS)
    buf = np.array(img, dtype="float32") / 255.0   # 1 = branco, 0 = preto
    n = len(RAMP)

    # niveis de "escuridao" 0..1 igualmente espacados; gamma aplica na ida
    def quant(v):
        v = min(1.0, max(0.0, v))
        dark = (1.0 - v) ** gamma
        idx = min(n - 1, int(dark * n))
        # valor representativo do nivel (centro do bucket) p/ calcular o erro
        rep = 1.0 - (idx + 0.5) / n
        if idx == 0:
            rep = 1.0
        elif idx == n - 1:
            rep = 1.0 - (n - 0.5) / n
        # desfaz o gamma aproximadamente
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
    """Monta o SVG com as duas camadas tematicas.

    lines_light desenha a sombra (fundo claro); lines_dark desenha a luz
    (fundo escuro). So uma camada e visivel por vez, via display + media
    query — o mesmo arquivo serve os dois esquemas do GitHub.
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
    # jitter por linha: desloca cada linha de ±JITTER*CHAR_W para quebrar o
    # alinhamento vertical — o pontilhado "tremido" do wallpaper. O pad extra
    # acomoda o deslocamento para a direita sem clipar. Mesmo seed nas duas
    # camadas, entao a geometria coincide.
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
                    help="pula o rembg e usa a foto inteira — use se o cabelo/"
                         "cabeca continuar sendo cortado")
    ap.add_argument("--no-dither", action="store_true",
                    help="desliga o Floyd-Steinberg (posteriza, menos detalhe)")
    ap.add_argument("--preview", action="store_true",
                    help="print the ASCII to the terminal as well")
    ap.add_argument("--out-txt", metavar="PATH",
                    help="also write the plain ASCII to this file, e.g. "
                         "~/.config/fastfetch/logo.txt")
    ap.add_argument("--out-txt-dark", metavar="PATH",
                    help="plain ASCII da camada dark (luz -> ponto)")
    ap.add_argument("--no-dark", action="store_true",
                    help="gera so a camada light (retrato single-theme)")
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
