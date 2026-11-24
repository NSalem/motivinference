"""House style for the paper's figures: page geometry, Arial, type sizes, line weights, colours, labels.

Every figure script (``scripts/exploratory/figures/``) calls ``figstyle.apply()`` once and is saved at its
final printed size, so 7 pt in the code is 7 pt on the page. Text sizes and line widths are never rescaled.

Layout vs page: the double-column figures are laid out in millimetres on a ``LAYOUT_DOUBLE`` = 183 mm page
and saved at the journal's ``DOUBLE`` width; ``page_scale()`` is the factor between the two (180/183 for NHB).
``Canvas`` (and fig1's own canvas) create the figure at layout size × ``page_scale``, so positions, panel sizes
and drawings shrink uniformly while type and line widths keep their point sizes.

Journal geometry, Nature Human Behaviour (nature.com/nathumbehav "AIP and formatting", checked 13 Nov 2026):
figures at most 180 mm wide, at least 300 dpi, text 5-7 pt sans serif; single column 89 mm and 170 mm maximum
height from the Nature Portfolio figure guide (checked 19 Oct 2026); Arial, live text, fonts embedded as Type 42.

Arial: registered from ``MOTINF_FONT_DIR``, the Windows font folder (WSL or native) or the macOS font
folders. Without Arial, glyphs are drawn with Liberation Sans, which has
Arial's metrics (same widths, so the same layout), and every SVG still names Arial as its font, so the
files show in Arial on any machine that has it. PDF export is refused without Arial, because the PDF
would embed Liberation Sans.
"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager

# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

SINGLE, MID, DOUBLE, MAX_HEIGHT = 89.0, 120.0, 180.0, 170.0  # mm (NHB)
LAYOUT_DOUBLE = 183.0  # page width (mm) the double-column layouts are drawn in; saved at DOUBLE


def page_scale(layout_width: float) -> float:
    """Factor from layout millimetres to printed millimetres: DOUBLE / LAYOUT_DOUBLE for a double-column layout,
    1 for any other width (single-column figures are laid out at their printed size)."""
    return DOUBLE / LAYOUT_DOUBLE if abs(layout_width - LAYOUT_DOUBLE) < 1e-9 else 1.0


def mm(*values: float):
    """Millimetres to inches (matplotlib's unit). ``mm(w, h)`` returns a figsize tuple."""
    out = tuple(v / 25.4 for v in values)
    return out[0] if len(out) == 1 else out


def letter(i: int) -> str:
    """Panel letter (lowercase, as NHB prints them)."""
    return "abcdefghijklmnopqrstuvwxyz"[i]


# --------------------------------------------------------------------------- #
# Type and lines (points, at final size)
# --------------------------------------------------------------------------- #

FONT = "Arial"
FALLBACK = "Liberation Sans"  # Arial's metrics; used only to draw glyphs when Arial is absent
SIZE_TEXT = 6.0  # tick labels, annotations, legends
SIZE_LABEL = 7.0  # axis labels, panel titles
SIZE_LETTER = 8.0  # panel letters (bold)
SIZE_MIN, SIZE_MAX = 5.0, 7.0  # journal limits for everything except panel letters

LW_AXIS = 0.5
LW_DATA = 0.75
LW_THIN = 0.4
TICK_LEN = 2.0

TEXT = "#000000"
MUTED = "#595959"  # secondary text: second axes, notes
REF_LINE = "#8C8C8C"  # zero lines, reference lines

# --------------------------------------------------------------------------- #
# Colours: the paper's original codes (decision 19 Oct 2026: keep red / grey / green)
# --------------------------------------------------------------------------- #

LOSS, NEUTRAL, GAIN = -1.0, 0.0, 1.0
INC_COLORS = {LOSS: "#e41a1c", NEUTRAL: "#8c8c8c", GAIN: "#4daf4a"}  # one red, grey and green everywhere

# Noise sources: the Fig. 3 set of the original code (Fig. 2 and Supp. Fig. 4 used lighter variants)
NOISE_COLORS = {"sen": "#b26f01", "inf": "#0a80a9", "sel": "#8c00b2"}
NOISE_COLORS_LIGHT = {"sen": "#febb4d", "inf": "#56cdf5", "sel": "#d84cff"}
BG_TINT = 0.15  # shaded panel backgrounds (Figs 2a, 3a): the colour at 15% on white, opaque (``tint``)

DOT_SHADE = 0.75  # participant dots: condition colour darkened by this factor (face only, no edge)
BOX_SHADE = 0.7  # box edges: condition colour darkened, as in the original code


def shade(color, factor: float):
    """Colour darkened by ``factor`` (0-1)."""
    r, g, b = mpl.colors.to_rgb(color)
    return (factor * r, factor * g, factor * b)


def tint(color, factor: float):
    """Colour mixed with white: ``factor`` = 1 keeps the colour, 0 gives white."""
    r, g, b = mpl.colors.to_rgb(color)
    return (1 - factor * (1 - r), 1 - factor * (1 - g), 1 - factor * (1 - b))


# --------------------------------------------------------------------------- #
# Labels (one place for every string shown in the figures)
# --------------------------------------------------------------------------- #

# Sensory-noise symbol: σ_ang, as in the manuscript text and tables.
SENSORY_SUBSCRIPT = "ang"

INC_TICKS = {LOSS: "−1", NEUTRAL: "0", GAIN: "+1"}  # decision 19 Oct 2026: keep −1 / 0 / +1
INC_XLABEL = "Incentive (£)"
EFF_XLABEL = "vs £0"
ABS_TICKS = {0.0: "0", 1.0: "±1"}
ABS_XLABEL = "|Incentive| (£)"
SEQ_LENGTH = "Sequence length"


def sym(kind: str, channel: str) -> str:
    """Mathtext symbol: ``sym("sigma", "inf")`` -> σ_inf; ``sym("delta", "sen")`` -> δ_ang."""
    sub = SENSORY_SUBSCRIPT if channel in ("sen", "ang") else channel
    greek = {"sigma": r"\sigma", "delta": r"\delta"}[kind]
    return rf"${greek}_{{\mathrm{{{sub}}}}}$"


# --------------------------------------------------------------------------- #
# Fonts
# --------------------------------------------------------------------------- #

_ARIAL_FILES = ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf",
                "Arial.ttf", "Arial Bold.ttf", "Arial Italic.ttf", "Arial Bold Italic.ttf")
HAS_ARIAL = False


def _font_dirs() -> list[Path]:
    dirs = [Path(os.environ["MOTINF_FONT_DIR"])] if os.environ.get("MOTINF_FONT_DIR") else []
    dirs += [Path("/mnt/c/Windows/Fonts"), Path("C:/Windows/Fonts"), Path("/Library/Fonts"),
             Path.home() / "Library/Fonts", Path("/usr/share/fonts/truetype/msttcorefonts")]
    return dirs


def _register_arial() -> bool:
    for d in _font_dirs():
        for f in _ARIAL_FILES:
            p = d / f
            if p.is_file():
                font_manager.fontManager.addfont(str(p))
    return any(f.name == FONT for f in font_manager.fontManager.ttflist)


# --------------------------------------------------------------------------- #
# rcParams
# --------------------------------------------------------------------------- #


def rc() -> dict:
    glyphs = FONT if HAS_ARIAL else FALLBACK
    return {
        "font.family": "sans-serif",
        "font.sans-serif": [FONT, FALLBACK, "DejaVu Sans"],
        "font.size": SIZE_TEXT,
        "axes.labelsize": SIZE_LABEL,
        "axes.titlesize": SIZE_LABEL,
        "axes.titleweight": "normal",
        "figure.titlesize": SIZE_LABEL,
        "xtick.labelsize": SIZE_TEXT,
        "ytick.labelsize": SIZE_TEXT,
        "legend.fontsize": SIZE_TEXT,
        "legend.title_fontsize": SIZE_TEXT,
        "legend.frameon": False,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.4,
        "legend.borderaxespad": 0.2,
        "mathtext.fontset": "custom",
        "mathtext.rm": glyphs,
        "mathtext.it": f"{glyphs}:italic",
        "mathtext.bf": f"{glyphs}:bold",
        "mathtext.sf": glyphs,
        "mathtext.cal": f"{glyphs}:italic",
        "mathtext.default": "it",
        "axes.unicode_minus": True,
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "axes.edgecolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "axes.linewidth": LW_AXIS,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelpad": 2.0,
        "axes.titlepad": 3.0,
        "xtick.major.width": LW_AXIS,
        "ytick.major.width": LW_AXIS,
        "xtick.major.size": TICK_LEN,
        "ytick.major.size": TICK_LEN,
        "xtick.major.pad": 1.5,
        "ytick.major.pad": 1.5,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "lines.linewidth": LW_DATA,
        "lines.markersize": 3.0,
        "lines.solid_capstyle": "butt",
        "patch.linewidth": LW_AXIS,
        "errorbar.capsize": 0.0,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": None,  # never "tight": the panel size must not depend on its labels
        "savefig.pad_inches": 0.0,
        "savefig.transparent": False,
        "figure.facecolor": "white",
        "svg.fonttype": "none",  # live, editable text
        "svg.hashsalt": "motinf",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.constrained_layout.h_pad": mm(0.5),
        "figure.constrained_layout.w_pad": mm(0.5),
        "figure.constrained_layout.hspace": 0.02,
        "figure.constrained_layout.wspace": 0.02,
    }


def apply() -> None:
    """Register Arial (if found) and set the house rcParams. Call once per script, before drawing."""
    global HAS_ARIAL
    HAS_ARIAL = _register_arial()
    if not HAS_ARIAL:
        warnings.warn(f"Arial not found; drawing with {FALLBACK} (same metrics). SVGs still name Arial; "
                      "PDF export is disabled.", stacklevel=2)
    mpl.rcParams.update(rc())


# --------------------------------------------------------------------------- #
# Checks and export
# --------------------------------------------------------------------------- #


def check(fig, name: str = "") -> list[str]:
    """Warn about text outside 5-7 pt (panel letters excepted), coloured text and figures wider than a
    double column or taller than the maximum height."""
    issues = []
    w, h = (25.4 * v for v in fig.get_size_inches())
    if w > DOUBLE + 0.5:
        issues.append(f"width {w:.1f} mm > {DOUBLE} mm")
    if h > MAX_HEIGHT + 0.5:
        issues.append(f"height {h:.1f} mm > {MAX_HEIGHT} mm")
    for t in fig.findobj(mpl.text.Text):
        s = t.get_text().strip()
        if not s or not t.get_visible():
            continue
        size = t.get_fontsize()
        is_letter = t.get_fontweight() in ("bold", 700) and len(s) == 1 and size == SIZE_LETTER
        if not is_letter and not (SIZE_MIN - 1e-6 <= size <= SIZE_MAX + 1e-6):
            issues.append(f"{size:.1f} pt: {s[:30]!r}")
        rgb = mpl.colors.to_rgb(t.get_color())
        if max(rgb) - min(rgb) > 0.05:
            issues.append(f"coloured text: {s[:30]!r}")
    issues += overlaps(fig)
    for i in issues:
        warnings.warn(f"[{name}] {i}", stacklevel=2)
    return issues


def overlaps(fig, tol_pt: float = 0.3) -> list[str]:
    """Text that collides with other text or runs off the page (seam check; draws the figure once)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    tol = tol_pt * fig.dpi / 72
    # tick labels that are not drawn (hidden axis, or a tick outside the view limits) still report "visible"
    skip = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                loc = tk.get_loc()
                if (not ax.axison or not axis.get_visible() or loc is None
                        or not (lo - 1e-9 * (abs(lo) + 1) <= loc <= hi + 1e-9 * (abs(hi) + 1))):
                    skip |= {id(tk.label1), id(tk.label2)}
    items = []
    for t in fig.findobj(mpl.text.Text):
        s = t.get_text().strip()
        if not s or not t.get_visible() or t.get_alpha() == 0 or id(t) in skip:
            continue
        try:
            bb = t.get_window_extent(r)
        except Exception:  # noqa: BLE001
            continue
        if bb.width <= 0 or bb.height <= 0:
            continue
        items.append((s, bb))
    out = []
    for s, bb in items:
        if bb.x0 < -tol or bb.y0 < -tol or bb.x1 > W + tol or bb.y1 > H + tol:
            out.append(f"off the page: {s[:30]!r}")
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i][1], items[j][1]
            if min(a.x1, b.x1) - max(a.x0, b.x0) > tol and min(a.y1, b.y1) - max(a.y0, b.y0) > tol:
                out.append(f"text overlap: {items[i][0][:25]!r} / {items[j][0][:25]!r}")
    return out


_SVG_LIBERATION = re.compile(r"font-family: '?Liberation Sans'?(?=[;\"])")


def save(fig, path: str | os.PathLike, *, pdf: bool = True, png: bool = True) -> Path:
    """Write ``<path>.svg`` (live text, Arial), plus a PDF when Arial is available and a 300 dpi PNG preview.

    The figure keeps exactly its ``figsize``: no tight bounding box.
    """
    path = Path(path).with_suffix("")
    path.parent.mkdir(parents=True, exist_ok=True)
    check(fig, path.name)
    svg = path.with_suffix(".svg")
    fig.savefig(svg)
    # glyphs drawn with the fallback are still Arial text in the file
    s = svg.read_text(encoding="utf-8")
    s = _SVG_LIBERATION.sub("font-family: 'Arial', 'Liberation Sans', sans-serif", s)
    svg.write_text(s, encoding="utf-8")
    other = sorted({f for f in re.findall(r"font-family: ([^;\"]+)", s) if "Arial" not in f})
    if other:  # e.g. mathtext symbols missing from the font fall back to Computer Modern (cmsy10)
        warnings.warn(f"[{path.name}] glyphs in fonts other than Arial: {other}", stacklevel=2)
    if pdf and HAS_ARIAL:
        fig.savefig(path.with_suffix(".pdf"))
    if png:
        fig.savefig(path.with_suffix(".png"), dpi=300)
    return svg


# --------------------------------------------------------------------------- #
# Task names and number formatting
# --------------------------------------------------------------------------- #

# Task names of the manuscript, in every figure (the code and data files still say estim(ation) / categ, inf)
TASK_REPRO = "Stimulus Reproduction"
TASK_CATEG = "Sequence Categorization"


def fmt_num(v: float, nd: int = 2) -> str:
    """Number with a true minus sign and no negative zero: ``fmt_num(-0.001)`` -> ``"0.00"``."""
    s = f"{v:.{nd}f}"
    if float(s) == 0:
        s = f"{0:.{nd}f}"
    return s.replace("-", "−")
