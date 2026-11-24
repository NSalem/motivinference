"""Supplementary Figure 6: full-model parameters and individual differences in choice variability.

Panels:
    a  choice-variability midpoint and slope against each noise parameter: participants, linear fit to the data
       (solid, 95% CI) and to the model's predictions (dashed)
    b  standardized coefficients of robust linear models with all three noise parameters, for the data and the
       model's predictions (± SE)
    c, d  the same for the incentive effects: change in midpoint and slope (±£1 − £0) against δ

Inputs (results/exploratory/merged/modeling/stats/full_model_stats/correlations/): param_behavior_main.csv,
param_behavior_diff.csv, param_behavior_correlations.csv (r, P and model R² of a and c) and rlm_coefs.csv.
Output: plots/exploratory/figures/sfig6.{svg,pdf,png}, at print size (180 mm wide).

Usage (from the repository root):
    python scripts/exploratory/figures/sfig6.py
"""

from __future__ import annotations

import figpaths  # noqa: F401  (paths first)

import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.transforms import offset_copy

from motinf.exploratory.figlib import figstyle as st
from motinf.exploratory.figlib.canvas import Canvas
from motinf.exploratory.figlib.dist import p_stars
from motinf.exploratory.figlib.source_data import SourceData

st.apply()
ROOT, OUT = figpaths.DATA_ROOT, figpaths.OUT_ROOT
SD = SourceData("sfig6", OUT)
CORR = ROOT / "results/exploratory/merged/modeling/stats/full_model_stats/correlations"
rlm = pd.read_csv(CORR / "rlm_coefs.csv")
cor = pd.read_csv(CORR / "param_behavior_correlations.csv", float_precision="round_trip").set_index(["y", "x"])
rlm = rlm[rlm["predictor"] != "const"]
NC = st.NOISE_COLORS
CH = ("ang", "inf", "sel")
KEY = {"ang": "sen", "inf": "inf", "sel": "sel"}  # channel -> figstyle noise key
SRC_NAME = {"ang": "Sensory", "inf": "Inference", "sel": "Selection"}
DASH = (0, (3, 1.5))

BLOCKS = [
    dict(csv="param_behavior_main.csv", x=lambda ch: f"sd_{ch}", sym="sigma", rlm="main",
         rows=[("intercept", "intercept_mod", "Variability midpoint"), ("slope", "slope_mod", "Variability slope")],
         title_a="Noise parameters and choice variability", title_b="Robust regression, all three"),
    dict(csv="param_behavior_diff.csv", x=lambda ch: f"motiv_{ch}", sym="delta", rlm="diff",
         rows=[("intercept_diff", "intercept_mod_diff", "Δ variability midpoint"),
               ("slope_diff", "slope_mod_diff", "Δ variability slope")],
         title_a="Incentive parameters and the change in choice variability (±£1 − £0)",
         title_b="Robust regression, all three"),
]

W = st.LAYOUT_DOUBLE  # layout mm; saved at st.DOUBLE wide (st.page_scale)
AW, AH = 28.0, 21.0  # scatter axes
BG_L, BG_R = 7.5, 3.0  # background padding left / right of the axes (left: room for the widest tick labels)
COLX = tuple(16.0 + k * (AW + BG_L + BG_R + 2.5) for k in range(3))  # scatter axes x; 2.5 mm between backgrounds
YLAB_GAP = 1.5  # right edge of the first column's y labels, this far left of its background (mm)
ROW_DY = AH + 5.0  # top of row 2 relative to row 1 (room for its r / R² line)
BLOCK_H = 72.0
H = 2 * BLOCK_H
c = Canvas(W, H)
N = {}
BELOW = []  # star labels under a negative coefficient (moved down half a line at the end)

for bi, blk in enumerate(BLOCKS):
    yt = 1.6 + bi * BLOCK_H
    df = pd.read_csv(CORR / blk["csv"])
    xcols = [blk["x"](ch) for ch in CH]
    N[blk["csv"]] = len(df)
    ycols = [y for row in blk["rows"] for y in row[:2]]
    SD.add(st.letter(2 * bi), df[[c_ for c_ in ["participant"] + xcols + ycols if c_ in df.columns]],
           "per participant: full-model parameters, choice-variability midpoint and slope (data) and the same predicted "
           "by the model (*_mod)", part="participants")
    corr_rows = []
    c.letter(2 * bi, 0.5, yt)
    c.title(4.5, yt + 0.2, blk["title_a"])
    yb0 = yt + 5.0  # background top
    ya = yb0 + 9.5  # first scatter row (column title, then the r / R² line)
    yb1 = ya + ROW_DY + AH + 8.0  # background bottom, below the x labels
    for k, ch in enumerate(CH):
        x0 = COLX[k]
        c.box(x0 - BG_L, yb0, AW + BG_L + BG_R, yb1 - yb0,
              fc=st.tint(NC[KEY[ch]], st.BG_TINT))
        c.text(x0 + AW / 2, yb0 + 2.4, SRC_NAME[ch], ha="center", va="center", fontweight="bold",
               fontsize=st.SIZE_LABEL, zorder=4)
    xr = {}
    for xc in xcols:
        v = df[xc].dropna()
        pad = 0.05 * (v.max() - v.min())
        xr[xc] = (v.min() - pad, v.max() + pad)
    for r, (ycol, ymod, ylab) in enumerate(blk["rows"]):
        yv = df[ycol].dropna()
        span = yv.max() - yv.min()
        ylim = (yv.min() - 0.05 * span, yv.max() + 0.05 * span)
        for k, (ch, xc) in enumerate(zip(CH, xcols)):
            ax = c.axes_mm(COLX[k], ya + r * ROW_DY, AW, AH, bg=False)
            d = df[[xc, ycol]].dropna()
            ax.scatter(d[xc], d[ycol], s=2.0, facecolors=["0.3"], edgecolors="none", linewidths=0, alpha=0.55,
                       zorder=3)
            sns.regplot(x=d[xc], y=d[ycol], ax=ax, scatter=False, ci=95, seed=0, color="0.15",
                        line_kws={"lw": 0.9, "zorder": 4})
            dm = df[[xc, ymod]].dropna()
            sns.regplot(x=dm[xc], y=dm[ymod], ax=ax, scatter=False, ci=95, seed=0, color=NC[KEY[ch]],
                        line_kws={"lw": 0.9, "ls": DASH, "zorder": 5})
            ax.axhline(0, color=st.REF_LINE, lw=st.LW_THIN, ls=(0, (3, 2)), zorder=1)
            ax.set_xlim(*xr[xc])
            ax.set_ylim(*ylim)
            ax.locator_params(axis="x", nbins=3)
            ax.locator_params(axis="y", nbins=4)
            ax.set_xlabel("")
            ax.set_ylabel(ylab if k == 0 else "")
            if k == 0:  # same x for every row of both blocks, just outside the shaded background
                ax.yaxis.set_label_coords(-(BG_L + YLAB_GAP) / AW, 0.5)
            if k:
                ax.tick_params(axis="y", labelleft=False)
            if r == 0:
                ax.tick_params(axis="x", labelbottom=False)
            else:
                ax.set_xlabel(st.sym(blk["sym"], ch))
            rr, pp, m2, n = cor.loc[(ycol, xc), ["pearson_r", "p", "model_r2", "n"]]
            corr_rows.append({"y": ycol, "x": xc, "pearson_r": rr, "p": pp, "model_r2": m2, "n": int(n)})
            ax.text(0.0, 1.03, f"r = {st.fmt_num(rr)}{p_stars(pp)}", transform=ax.transAxes, ha="left",
                    va="bottom", fontsize=st.SIZE_TEXT)
            ax.text(1.0, 1.03, f"model $R^2$ = {st.fmt_num(m2)}", transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=st.SIZE_TEXT)

    SD.add(st.letter(2 * bi), corr_rows, "Pearson r of the data (with p) and R² of the data around the model's line, "
           "per plot", part="correlations")
    # b / d  robust regression coefficients
    xb = 134.0
    rlm_rows = []
    c.letter(2 * bi + 1, xb - 6.0, yt)
    c.title(xb - 2.0, yt + 0.2, blk["title_b"])
    for r, (ycol, ymod, ylab) in enumerate(blk["rows"]):
        ax = c.axes_mm(xb + 9.0, ya + r * ROW_DY, 37.0, AH)
        pred = [blk["x"](ch) for ch in CH]
        dd = rlm[(rlm["block"] == blk["rlm"]) & (rlm["y"] == ycol)].set_index("predictor").loc[pred]
        dm = rlm[(rlm["block"] == blk["rlm"]) & (rlm["y"] == ymod)].set_index("predictor").loc[pred]
        rlm_rows += [pd.concat([dd.assign(fitted_to="data"), dm.assign(fitted_to="model predictions")]).reset_index()]
        tops = []
        for k, ch in enumerate(CH):
            for off, row, model in ((-0.17, dd.iloc[k], False), (0.17, dm.iloc[k], True)):
                col = NC[KEY[ch]] if model else "k"
                ax.errorbar(k + off, row["coef"], yerr=row["se"], fmt="o", ms=3.0, color=col,
                            mfc="white" if model else col, mec=col, mew=0.7, elinewidth=st.LW_DATA, zorder=3)
                tops.append((k + off, row["coef"], row["se"], row["pvalue"]))
        ax.axhline(0, color=st.REF_LINE, lw=st.LW_THIN, ls=(0, (3, 2)), zorder=1)
        lo = min(min(v - s, 0) for _, v, s, _ in tops)
        hi = max(max(v + s, 0) for _, v, s, _ in tops)
        span = hi - lo
        ax.set_ylim(lo - 0.12 * span, hi + 0.22 * span)
        for x, v, s, p in tops:
            st_ = p_stars(p)
            if st_:
                t = ax.text(x, (v + s if v >= 0 else v - s) + (0.03 if v >= 0 else -0.03) * span, st_, ha="center",
                            va="bottom" if v >= 0 else "top", fontsize=st.SIZE_MIN)
                if v < 0:  # an asterisk sits high in its line: half a line lower, unless that reaches the bottom spine
                    BELOW.append((ax, t))
        ax.set_xlim(-0.5, 2.5)
        ax.set_xticks([0, 1, 2], [st.sym(blk["sym"], ch) for ch in CH] if r == 1 else ["", "", ""])
        ax.tick_params(axis="x", length=0)
        ax.locator_params(axis="y", nbins=4)
        ax.set_ylabel("Std. coefficient", labelpad=1.5)
        ax.text(0.5, 1.03, ylab, transform=ax.transAxes, ha="center", va="bottom", fontsize=st.SIZE_TEXT)
    SD.add(st.letter(2 * bi + 1), pd.concat(rlm_rows, ignore_index=True),
           "standardized coefficients (± SE, p) of robust linear models with all three parameters, fitted to the data "
           "and to the model's predictions")

c.fig.canvas.draw()
for ax, t in BELOW:  # stars under negative coefficients: half a line lower, unless that reaches the bottom spine
    plain = t.get_transform()
    t.set_transform(offset_copy(plain, fig=c.fig, y=-0.5 * 1.2 * st.SIZE_MIN, units="points"))
    if t.get_window_extent(c.fig.canvas.get_renderer()).y0 < ax.get_window_extent().y0 + 0.5:
        t.set_transform(plain)
        print("star kept in place (would reach the bottom spine):", repr(t.get_text()))

# one key for the whole figure, on the title line of a
hs = [Line2D([], [], marker="o", ms=2.6, lw=0, mfc="0.3", mec="none", label="participants"),
      Line2D([], [], color="0.15", lw=0.9, label="fit to data"),
      Line2D([], [], color="0.45", lw=0.9, ls=DASH, label="fit to model predictions (noise color)"),
      Line2D([], [], marker="o", ms=3.0, color="k", lw=0, label="data"),
      Line2D([], [], marker="o", ms=3.0, mfc="white", mec="0.45", mew=0.7, lw=0, label="model")]
c.fig.legend(handles=hs[:3], loc="upper right", ncol=3, bbox_to_anchor=(126.0 / W, 1 - 0.4 / H), borderaxespad=0,
             handlelength=1.8, columnspacing=1.0)
c.fig.legend(handles=hs[3:], loc="upper right", ncol=2, bbox_to_anchor=(182.5 / W, 1 - (1.6 + 5.2) / H),
             borderaxespad=0, handlelength=1.0, columnspacing=1.0)

st.save(c.fig, OUT / "figures" / "sfig6")
SD.write()
print("saved sfig6", f"{W:.0f} x {H:.0f} mm", N, "rlm n:", rlm.groupby("block")["n"].unique().to_dict())
