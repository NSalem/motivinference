"""Supplementary Figure 3: regressions of the noise parameters on questionnaires and demographics.

Robust linear regressions. Columns: reproduction error SD (σang), σinf, and their incentive effects for loss and
gain (δ = log σ£0 − log σincentive; positive: less noise under incentives). Rows: one regression per block of
predictors (BIS/BAS, SPSRQ-RC, age and sex), coefficient ± SE; stars, uncorrected p. Bottom: weighted R² of each
regression.

Inputs (results/exploratory/merged/modeling/stats/questionnaire_model_correlations/simple_model/core/):
rlm/rlm_coefs_long.csv and demographics/rlm/rlm_coefs_long.csv.
Output: plots/exploratory/figures/sfig3.{svg,pdf,png}, at print size (180 mm wide).

Usage (from the repository root):
    python scripts/exploratory/figures/sfig3.py
"""

from __future__ import annotations

import figpaths  # noqa: F401  (paths first)

import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

from motinf.exploratory.figlib import figstyle as st
from motinf.exploratory.figlib.canvas import Canvas
from motinf.exploratory.figlib.dist import p_stars
from motinf.exploratory.figlib.source_data import SourceData

st.apply()
ROOT, OUT = figpaths.DATA_ROOT, figpaths.OUT_ROOT
B = ROOT / "results/exploratory/merged/modeling/stats/questionnaire_model_correlations/simple_model/core"
df = pd.concat([pd.read_csv(B / "rlm/rlm_coefs_long.csv"), pd.read_csv(B / "demographics/rlm/rlm_coefs_long.csv")])
LOSS, GAIN = st.LOSS, st.GAIN

BLOCKS = [  # (block id, header, [(predictor, label)])
    ("bis_bas", "BIS/BAS", [("bis", "BIS"), ("bas_drv", "BAS Drive"), ("bas_rew", "BAS Reward Responsiveness"),
                            ("bas_fun", "BAS Fun Seeking")]),
    ("spsrq", "SPSRQ-RC", [("spsrqrc_sp", "Sensitivity to Punishment"), ("spsrqrc_sr", "Sensitivity to Reward")]),
    ("demographics", "Demographics", [("age", "Age (z-scored)"), ("sex_male", "Sex (male = 1)")]),
]
COLS = [  # (title, {series: parameter}); series None = single estimate (black)
    (st.sym("sigma", "ang"), {None: "sd_ang"}),
    (st.sym("sigma", "inf"), {None: "sd_inf"}),
    (st.sym("delta", "ang"), {LOSS: "motiv_ang_l", GAIN: "motiv_ang_g"}),
    (st.sym("delta", "inf"), {LOSS: "motiv_inf_l", GAIN: "motiv_inf_g"}),
]
DODGE = {None: 0.0, LOSS: -0.2, GAIN: 0.2}


def n_of(block):
    v = df.loc[df["block"] == block, "n"].unique()
    assert len(v) == 1, (block, v)
    return int(v[0])


def r2_of(block, param):
    v = df.loc[(df["block"] == block) & (df["parameter"] == param), "r2_full"].unique()
    assert len(v) == 1
    return float(v[0])


# row layout (data units, downwards): header line, then one line per predictor; 0.5 gap between blocks
ypos, heads, y = {}, {}, 0.0
for k, (bid, _, preds) in enumerate(BLOCKS):
    y += 0.5 if k else 0.0
    heads[bid] = y
    y += 1.0
    for p, _ in preds:
        ypos[(bid, p)] = y
        y += 1.0
Y1 = y - 0.4
r2pos, y = {}, 1.0
for bid, _, _ in BLOCKS:
    r2pos[bid] = y
    y += 1.0
Y2 = y - 0.4

W = st.LAYOUT_DOUBLE  # layout mm; saved at st.DOUBLE wide (st.page_scale)
PITCH = 4.0
TOP, X0, CW, GAP = 9.0, 46.0, 30.0, 4.3
H1 = (Y1 + 0.6) * PITCH  # axes span: ylim (Y1, -0.6)
TOP2 = TOP + H1 + 9.5
H2 = (Y2 - 0.4) * PITCH  # axes span: ylim (Y2, 0.4)
H = TOP2 + H2 + 8.6  # room for the R² superscript and descender of the bottom x labels
c = Canvas(W, H)

r2max = max(r2_of(b, p) for b, _, _ in BLOCKS for _, ps in COLS for p in ps.values())
R2LIM = (0, 1.25 * r2max)
axes1, axes2 = [], []
for j, (title, series) in enumerate(COLS):
    x = X0 + j * (CW + GAP)
    ax = c.axes_mm(x, TOP, CW, H1)
    axes1.append(ax)
    sub = df[df["parameter"].isin(series.values())]
    lim = 1.18 * float((sub["coef"].abs() + sub["se"]).max())
    for s_, param in series.items():
        col = "k" if s_ is None else st.INC_COLORS[s_]
        for bid, _, preds in BLOCKS:
            for p, _ in preds:
                r = df[(df["block"] == bid) & (df["parameter"] == param) & (df["predictor"] == p)]
                assert len(r) == 1, (bid, param, p)
                r = r.iloc[0]
                yy = ypos[(bid, p)] + DODGE[s_]
                ax.plot([r["coef"] - r["se"], r["coef"] + r["se"]], [yy, yy], color=col, lw=st.LW_DATA,
                        solid_capstyle="butt", zorder=3)
                ax.plot([r["coef"]], [yy], "o", ms=2.8, mfc=col, mec="white", mew=0.4, zorder=4)
                s = p_stars(r["pvalue"])
                if s:
                    right = r["coef"] >= 0
                    ax.text(r["coef"] + (1 if right else -1) * (r["se"] + 0.03 * lim), yy + 0.12, s,
                            ha="left" if right else "right", va="center", fontsize=st.SIZE_TEXT, zorder=4)
    ax.axvline(0, color=st.REF_LINE, lw=st.LW_THIN, ls=(0, (3, 2)), zorder=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(Y1, -0.6)
    for bid, _, _ in BLOCKS[1:]:
        ax.axhline(heads[bid] - 0.25, color="0.85", lw=st.LW_THIN, zorder=0)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.locator_params(axis="x", nbins=3)
    ax.set_xlabel("Coefficient ± SE")
    c.text(x + CW / 2, TOP - 1.0, title, ha="center", va="bottom", fontsize=st.SIZE_LABEL)

    # weighted R² of each regression
    ax2 = c.axes_mm(x, TOP2, CW, H2)
    axes2.append(ax2)
    for s_, param in series.items():
        col = "0.55" if s_ is None else st.INC_COLORS[s_]
        hgt = 0.62 if s_ is None else 0.36
        for bid, _, _ in BLOCKS:
            v = r2_of(bid, param)
            yy = r2pos[bid] + DODGE[s_]
            ax2.barh(yy, v, height=hgt, color=st.tint(col, 0.55) if s_ is None else st.tint(col, 0.7), ec="none",
                     zorder=2)
            ax2.text(v + 0.012 * R2LIM[1] * 4, yy, st.fmt_num(v), ha="left", va="center", fontsize=st.SIZE_MIN)
    ax2.set_xlim(*R2LIM)
    ax2.set_ylim(Y2, 0.4)
    ax2.set_yticks([])
    ax2.spines["left"].set_bounds(Y2 - 0.1, 0.5)
    ax2.set_xticks([0, 0.05, 0.1], ["0", "0.05", "0.10"])
    ax2.set_xlabel(r"Weighted $R^2$")

# row labels and block headers in the label column
tr = blended_transform_factory(axes1[0].transAxes, axes1[0].transData)
for bid, head, preds in BLOCKS:
    axes1[0].text(-(X0 - 1.0) / CW, heads[bid], f"{head} (n = {n_of(bid)})", transform=tr, ha="left",
                  va="center", fontsize=st.SIZE_TEXT, fontweight="bold")
    for p, lab in preds:
        axes1[0].text(-1.2 / CW, ypos[(bid, p)], lab, transform=tr, ha="right", va="center", fontsize=st.SIZE_TEXT)
tr2 = blended_transform_factory(axes2[0].transAxes, axes2[0].transData)
axes2[0].text(-(X0 - 1.0) / CW, 0.0, r"Variance explained", transform=tr2, ha="left", va="center",
              fontsize=st.SIZE_TEXT, fontweight="bold")
for bid, head, _ in BLOCKS:
    axes2[0].text(-1.2 / CW, r2pos[bid], head, transform=tr2, ha="right", va="center", fontsize=st.SIZE_TEXT)

# key for the incentive-effect columns
hs = [Line2D([], [], marker="o", ms=2.8, mfc=st.INC_COLORS[i], mec="white", mew=0.4, color=st.INC_COLORS[i],
             lw=st.LW_DATA, label=st.INC_TICKS[i]) for i in (LOSS, GAIN)]
c.fig.legend(handles=hs, title=st.INC_XLABEL, loc="upper left", ncol=2, alignment="left",
             bbox_to_anchor=(0.3 / W, 1 - 0.3 / H), handlelength=1.4, columnspacing=1.0,
             borderaxespad=0, title_fontsize=st.SIZE_TEXT)

SD = SourceData("sfig3", OUT)
plotted = df[df["parameter"].isin([p for _, s_ in COLS for p in s_.values()])
             & df["predictor"].isin([p for _, _, preds in BLOCKS for p, _ in preds])]
SD.add("", plotted[[c_ for c_ in ("block", "parameter", "predictor", "coef", "se", "pvalue", "n", "r2_full")
                    if c_ in plotted.columns]],
       "robust linear regressions of the noise parameters and their incentive effects (_l: loss, _g: gain) on each "
       "block of predictors: coefficient, SE, uncorrected p, n and weighted R² of the regression")
st.save(c.fig, OUT / "figures" / "sfig3")
SD.write()
print("saved sfig3", f"{W:.0f} x {H:.1f} mm", {b: n_of(b) for b, _, _ in BLOCKS})
