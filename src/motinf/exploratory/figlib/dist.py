"""Distribution panels at print scale (the figure scripts' replacement for the drawing part of ``plot_effects``).

Per condition: a narrow box (IQR, median, whiskers to 1.5 IQR) with the mean ± 95% CI in black on
it, and to its right a half violin (outline + light fill) holding the participants as dots (darker
face, no edge). Effects (incentive − neutral) use the same layout, one column per contrast, with a
dashed zero line. Significance marks: *, ** and *** for p < .05, .01 and .001 (no "ns"), in black;
values beyond the y limits: a small grey triangle at the edge with the count.

The statistics (means, CIs, KDE, robust limits) and ``log_axis`` are those of the earlier panel scripts, unchanged.
All sizes are in points at final size; call ``figstyle.apply()`` first.
"""

from __future__ import annotations

from typing import Any, Sequence

import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, NullLocator
from scipy import stats

from motinf.exploratory.figlib import figstyle as st

LOSS, NEUTRAL, GAIN = st.LOSS, st.NEUTRAL, st.GAIN


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def finite(v: Any) -> np.ndarray:
    v = np.asarray(v, dtype=float).ravel()
    return v[np.isfinite(v)]


def t_ci(v: Any, conf: float = 0.95) -> tuple[float, float, float]:
    """Mean and t-based CI of the mean."""
    v = finite(v)
    n = v.size
    m = float(v.mean())
    h = float(stats.t.ppf(0.5 + conf / 2, n - 1) * v.std(ddof=1) / np.sqrt(n))
    return m, m - h, m + h


def within_ci(x: np.ndarray, conf: float = 0.95) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cousineau-Morey within-participant CI of each column mean (rows = participants)."""
    x = np.asarray(x, dtype=float)
    n, k = x.shape
    norm = x - x.mean(axis=1, keepdims=True) + x.mean()
    se = norm.std(axis=0, ddof=1) * np.sqrt(k / (k - 1)) / np.sqrt(n)
    h = stats.t.ppf(0.5 + conf / 2, n - 1) * se
    m = x.mean(axis=0)
    return m, m - h, m + h


def condition_ci(x: np.ndarray, ci: str = "between") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean ± 95% CI per column: of each condition mean (``between``) or Cousineau-Morey (``within``)."""
    if ci == "within":
        return within_ci(x)
    m, lo, hi = zip(*(t_ci(x[:, j]) for j in range(x.shape[1])))
    return np.array(m), np.array(lo), np.array(hi)


def _kde(t: np.ndarray, n_grid: int = 256):
    """Gaussian KDE with a robust (Silverman) bandwidth, cut at the data range, peak scaled to 1."""
    sd = t.std(ddof=1)
    iqr = np.subtract(*np.percentile(t, [75, 25]))
    s = min(sd, iqr / 1.349) if iqr > 0 else sd
    k = stats.gaussian_kde(t, bw_method=0.9 * s * t.size ** -0.2 / sd)
    y = np.linspace(t.min(), t.max(), n_grid)
    d = k(y)
    return y, d / d.max(), k, d.max()


def robust_limits(values: Sequence[Any], k: float = 3.0, pad: float = 0.06, include=(0.0,)) -> tuple[float, float]:
    """Data range clipped to quartiles ± ``k`` IQR (pooled over ``values``), widened to ``include``."""
    allv = np.concatenate([finite(v) for v in values])
    q1, q3 = np.percentile(allv, [25, 75])
    iqr = q3 - q1
    lo, hi = max(allv.min(), q1 - k * iqr), min(allv.max(), q3 + k * iqr)
    for z in include:
        lo, hi = min(lo, z), max(hi, z)
    span = hi - lo
    return lo - pad * span, hi + pad * span


def log_axis(ax, ticks: Sequence[float]):
    ax.set_yscale("log")
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticks(list(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

# glyphs (points)
VIOLIN_LW = 0.5
VIOLIN_ALPHA = 0.18
DOT_SIZE = 1.6  # marker area, pt^2 (about 1.4 pt across)
DOT_ALPHA = 0.7  # face only; the edge is switched off explicitly
BOX_WIDTH = 0.16  # x units
BOX_LW = 0.6
WHISKER_LW = 0.5
MEDIAN_LW = 0.9
MEAN_MS = 3.2
CI_LW = 1.0
HALO = 0.6
MEANLINE_LW = 0.6


def p_stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def half_violin(ax, x, v, *, side=1, width=0.32, color, fill=True, log=False, points=True, rng=None,
                zorder=2.0):
    v = finite(v)
    t = np.log(v) if log else v
    y, d, k, dmax = _kde(t)
    yy = np.exp(y) if log else y
    xs = x + side * width * d
    if fill:
        ax.fill_betweenx(yy, x, xs, facecolor=color, alpha=VIOLIN_ALPHA, lw=0, zorder=zorder)
    ax.plot(xs, yy, color=color, lw=VIOLIN_LW, zorder=zorder + 0.1)
    if points:
        rng = rng or np.random.default_rng(0)
        off = side * width * (k(t) / dmax) * rng.uniform(0.08, 0.92, size=t.size)
        ax.scatter(x + off, v, s=DOT_SIZE, facecolors=[st.shade(color, st.DOT_SHADE)], edgecolors="none",
                   linewidths=0, alpha=DOT_ALPHA, zorder=zorder + 0.5)


def box(ax, x, v, *, color, width=BOX_WIDTH, whis=1.5, log=False, zorder=4.0):
    """Narrow box: IQR, median, whiskers to the last values within ``whis`` IQR (no fliers)."""
    v = finite(v)
    t = np.log(v) if log else v
    q1, med, q3 = np.percentile(t, [25, 50, 75])
    iqr = q3 - q1
    lo, hi = t[t >= q1 - whis * iqr].min(), t[t <= q3 + whis * iqr].max()
    f = np.exp if log else (lambda z: z)
    edge = st.shade(color, st.BOX_SHADE)
    ax.plot([x, x], [f(lo), f(q1)], color=edge, lw=WHISKER_LW, zorder=zorder, solid_capstyle="butt")
    ax.plot([x, x], [f(q3), f(hi)], color=edge, lw=WHISKER_LW, zorder=zorder, solid_capstyle="butt")
    ax.add_patch(Rectangle((x - width / 2, f(q1)), width, f(q3) - f(q1), facecolor="white", edgecolor=edge,
                           lw=BOX_LW, zorder=zorder + 0.01))
    ax.plot([x - width / 2, x + width / 2], [f(med), f(med)], color=edge, lw=MEDIAN_LW, zorder=zorder + 0.02,
            solid_capstyle="butt")


def mean_ci(ax, x, m, lo, hi, *, color="k", zorder=6.0):
    halo = [pe.Stroke(linewidth=CI_LW + 2 * HALO, foreground="white"), pe.Normal()]
    ax.plot([x, x], [lo, hi], color=color, lw=CI_LW, zorder=zorder, path_effects=halo,
            solid_capstyle="butt")
    ax.plot([x], [m], "o", ms=MEAN_MS, mfc=color, mec="white", mew=HALO, zorder=zorder + 0.01)


def zero_line(ax):
    ax.axhline(0.0, color=st.REF_LINE, lw=st.LW_THIN, ls=(0, (3, 2)), zorder=0.5)


def stars_at(ax, x, p):
    s = p_stars(p)
    if s:
        ax.text(x, 1.0, s, transform=ax.get_xaxis_transform(), ha="center", va="bottom", clip_on=False,
                fontsize=st.SIZE_TEXT, color=st.TEXT)


def mark_out_of_range(ax, x, v, lo, hi):
    v = finite(v)
    span = hi - lo
    for n, y, mk, va in (((v < lo).sum(), lo + 0.03 * span, "v", "bottom"),
                         ((v > hi).sum(), hi - 0.03 * span, "^", "top")):
        if n:
            ax.plot([x], [y], marker=mk, ms=2.2, color=st.MUTED, lw=0, zorder=8, clip_on=False)
            ax.annotate(str(int(n)), (x, y), xytext=(2.0, 0), textcoords="offset points", color=st.MUTED,
                        fontsize=st.SIZE_MIN, va="center", ha="left", zorder=8, annotation_clip=False)


def spine_to_ticks(ax):
    lo, hi = ax.get_xlim()
    t = [x for x in ax.get_xticks() if lo <= x <= hi]
    if t:
        ax.spines["bottom"].set_bounds(min(t), max(t))


def pct_axis(ax, kind: str = "change", label: str | None = None):
    """Right axis in % for a log-ratio axis: ``change`` 100(e^d − 1), ``reduction`` 100(1 − e^−d)."""
    inv = (lambda p: np.log1p(p / 100)) if kind == "change" else (lambda p: -np.log1p(-p / 100))
    lo, hi = ax.get_ylim()
    ticks = [0] if lo <= 0 <= hi else []
    for seq in ((10, 25, 50, 100, 200), (-10, -25, -50, -75, -90)):
        last = 0.0
        for p in seq:
            with np.errstate(divide="ignore", invalid="ignore"):  # a reduction of ≥ 100%: inf / nan, never a tick
                d = float(inv(p))
            if lo <= d <= hi and abs(d - last) >= 0.18 * (hi - lo):
                ticks.append(p)
                last = d
    ticks.sort()
    tw = ax.twinx()
    tw.set_ylim(lo, hi)
    tw.set_yticks([inv(p) for p in ticks], [f"{p:g}" for p in ticks])
    tw.set_ylabel(label or ("% change" if kind == "change" else "% reduction"), color=st.MUTED)
    tw.tick_params(axis="y", colors=st.MUTED)
    tw.spines["right"].set_visible(True)
    tw.spines["right"].set_color(st.MUTED)
    for side in ("left", "top", "bottom"):
        tw.spines[side].set_visible(False)
    return tw


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #


def levels_panel(ax, wide: pd.DataFrame, *, levels=(LOSS, NEUTRAL, GAIN), colors=None, names=None, log=False,
                 points=True, boxes=True, ci="between", width=0.55, gap=0.12, ylabel=None,
                 xlabel=st.INC_XLABEL, rng_seed=0, x0=0.0, fill=True, finish=True):
    """Per-condition distributions (``wide``: participants × levels) and mean ± 95% CI, means joined.

    ``x0`` shifts the group (several groups in one axes); ``fill=False`` draws the violins as outlines;
    ``finish=False`` leaves ticks, limits and labels to the caller."""
    colors = colors or st.INC_COLORS
    names = names or st.INC_TICKS
    X = wide[list(levels)].dropna().to_numpy(float)
    m, lo, hi = condition_ci(np.log(X) if log else X, ci)
    if log:
        m, lo, hi = np.exp(m), np.exp(lo), np.exp(hi)
    xs = x0 + np.arange(len(levels), dtype=float)
    rng = np.random.default_rng(rng_seed)
    for i, lvl in enumerate(levels):
        half_violin(ax, xs[i] + gap, X[:, i], width=width, color=colors[lvl], fill=fill, log=log, points=points,
                    rng=rng)
        if boxes:
            box(ax, xs[i], X[:, i], color=colors[lvl], log=log)
    ax.plot(xs, m, color="k", lw=MEANLINE_LW, zorder=5.5)
    for i in range(len(levels)):
        mean_ci(ax, xs[i], m[i], lo[i], hi[i])
    if finish:
        ax.set_xticks(xs, [names[lvl] for lvl in levels])
        ax.set_xlim(xs[0] - 0.35, xs[-1] + gap + width + 0.05)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        spine_to_ticks(ax)
    return xs


def effects_panel(ax, effects: dict[float, Any], *, contrasts=(LOSS, GAIN), colors=None, names=None,
                  pvals=None, points=True, boxes=True, width=0.55, gap=0.12, robust_k=3.0, ylim=None, ylabel=None,
                  xlabel=st.EFF_XLABEL, rng_seed=0, x0=0.0, fill=True, finish=True):
    """Per-participant effects, one column per contrast, drawn like ``levels_panel``: mean ± 95% CI of
    the paired difference with the distribution to its right; zero line; stars (``pvals``: ``{contrast: p}``).

    ``x0``, ``fill`` and ``finish`` as in ``levels_panel``; with several groups in one axes, pass the same
    ``ylim`` to each call."""
    colors = colors or st.INC_COLORS
    names = names or st.INC_TICKS
    pvals = pvals or {}
    rng = np.random.default_rng(rng_seed)
    keys = [c for c in contrasts if c in effects]
    xs = x0 + np.arange(len(keys), dtype=float)
    vals = {c: finite(effects[c]) for c in keys}
    for x, c in zip(xs, keys):
        half_violin(ax, x + gap, vals[c], width=width, color=colors[c], fill=fill, points=points, rng=rng)
        if boxes:
            box(ax, x, vals[c], color=colors[c])
        mean_ci(ax, x, *t_ci(vals[c]))
    lo, hi = ylim if ylim is not None else robust_limits(list(vals.values()), k=robust_k)
    ax.set_ylim(lo, hi)
    for x, c in zip(xs, keys):
        stars_at(ax, x, pvals.get(c, np.nan))
        mark_out_of_range(ax, x + gap + 0.5 * width, vals[c], lo, hi)
    if finish:
        zero_line(ax)
        ax.set_xticks(xs, [names[c] for c in keys])
        ax.set_xlim(xs[0] - 0.35, xs[-1] + gap + width + 0.05)
        if ylabel:
            ax.set_ylabel(ylabel)
        if xlabel:
            ax.set_xlabel(xlabel)
        spine_to_ticks(ax)
    return xs


def incentive_effects(wide: pd.DataFrame, scale: float = 1.0) -> dict[float, np.ndarray]:
    w = wide.dropna()
    return {c: scale * (w[c] - w[NEUTRAL]).to_numpy(float) for c in (LOSS, GAIN)}
