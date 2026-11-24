"""Supplementary Figure 1: incentive effects on every measure, in each experiment and pooled.

Standardized effects (dz with exact 95% CI) of loss − neutral and gain − neutral, for every measure of Tables 2
and 3 and for median response time: Experiment 1 (open circles), Experiment 2 (filled circles), pooled (black
diamonds). Stars: Holm-corrected p over the three contrasts of a measure, as in the tables. Noise and bias SDs on
the log scale, as in Table 3.

Input: results/exploratory/merged/paper_tables/paper_tables.json (table2).
Output: plots/exploratory/figures/sfig1.{svg,pdf,png}, at print size (180 mm wide).

Usage (from the repository root):
    python scripts/exploratory/figures/sfig1.py
"""

from __future__ import annotations

import json

import figpaths  # noqa: F401  (paths first)

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

from motinf.exploratory.figlib import figstyle as st
from motinf.exploratory.figlib.canvas import Canvas
from motinf.exploratory.figlib.dist import p_stars
from motinf.exploratory.figlib.source_data import SourceData

st.apply()
ROOT, OUT = figpaths.DATA_ROOT, figpaths.OUT_ROOT
T2 = json.loads((ROOT / "results/exploratory/merged/paper_tables/paper_tables.json").read_text())["table2"]

ROWS = [  # (task, label, key in table2[sample]); order and labels of Tables 2-3
    (st.TASK_CATEG, "Evidence sensitivity", "sensitivity"),
    (st.TASK_CATEG, "Optimal choices", "categ_optim"),
    (st.TASK_CATEG, st.sym("sigma", "inf") + " (log)", "log_sd_inf"),
    (st.TASK_CATEG, "Noise SD (log)", "categ_noise_log"),
    (st.TASK_CATEG, "Bias SD (log)", "categ_bias"),
    (st.TASK_CATEG, "Median response time", "categ_rt"),
    (st.TASK_REPRO, "Error SD", "estim_sd"),
    (st.TASK_REPRO, "Noise SD (log)", "estim_noise_log"),
    (st.TASK_REPRO, "Bias SD (log)", "estim_bias"),
    (st.TASK_REPRO, "Median response time", "estim_rt"),
]
SAMPLES = [("exp1", "Exp. 1", "o"), ("exp2", "Exp. 2", "o"), ("merged", "Pooled", "D")]
CONTRASTS = [("loss_neutral", st.LOSS, "Loss − neutral"), ("gain_neutral", st.GAIN, "Gain − neutral")]
N = {s: {T2[s][k][c]["n"] for *_, k in ROWS for c, *_ in CONTRASTS} for s, *_ in SAMPLES}
assert all(len(v) == 1 for v in N.values()), N
N = {s: v.pop() for s, v in N.items()}

# row positions (data units, increasing downwards); a 0.6 gap between the two task blocks
ys, y = [], 0.0
for i, (task, _, _) in enumerate(ROWS):
    if i and task != ROWS[i - 1][0]:
        y += 0.6
    ys.append(y)
    y += 1.0
ys = np.array(ys)
BLOCK = {t: [yy for yy, r in zip(ys, ROWS) if r[0] == t] for t in (st.TASK_CATEG, st.TASK_REPRO)}
YLIM = (ys.max() + 0.6, -0.6)
STAR = (6, 2, 0)  # six-armed asterisk
STAR_X0, STAR_DX = 0.035, 0.021  # first asterisk right of the CI, then one per level (dz units; ~0.9 mm apart)
OFF = {"exp1": -0.28, "exp2": 0.0, "merged": 0.28}


def star_xs(r: dict) -> list[float]:
    """x of the asterisks after a CI: *, ** or *** for Holm p < .05, .01, .001 (the tables' symbols)."""
    return [r["dz_hi"] + STAR_X0 + i * STAR_DX for i in range(len(p_stars(r["p_holm"])))]


res = [T2[s][k][c] for s, *_ in SAMPLES for *_, k in ROWS for c, *_ in CONTRASTS]
# right limit: room for the widest CI and its asterisks (at least 0.92)
X_RIGHT = max([r["dz_hi"] + 0.06 for r in res] + [x + 0.025 for r in res for x in star_xs(r)])
XLIM = (-0.62, max(0.92, float(np.ceil(X_RIGHT * 50) / 50)))
assert XLIM[0] < min(r["dz_lo"] for r in res)

W = st.LAYOUT_DOUBLE  # layout mm; saved at st.DOUBLE wide (st.page_scale)
AX_TOP, ROW_MM = 9.0, 6.4
AX_H = (YLIM[0] - YLIM[1]) * ROW_MM
H = AX_TOP + AX_H + 8.3
c = Canvas(W, H)
X0, AW, GAP = 41.0, 69.0, 3.5
axes = []
for j, (ckey, inc, title) in enumerate(CONTRASTS):
    ax = c.axes_mm(X0 + j * (AW + GAP), AX_TOP, AW, AX_H, bg=False)
    axes.append(ax)
    col = st.INC_COLORS[inc]
    cat = BLOCK[st.TASK_CATEG]
    ax.axhspan(min(cat) - 0.5, max(cat) + 0.5, color="0.94", lw=0, zorder=0)
    for yy, (_, _, key) in zip(ys, ROWS):
        for s, _, mk in SAMPLES:
            r = T2[s][key][ckey]
            yk = yy + OFF[s]
            pooled = s == "merged"
            lc = "k" if pooled else col
            ax.plot([r["dz_lo"], r["dz_hi"]], [yk, yk], color=lc, lw=1.0 if pooled else st.LW_DATA,
                    solid_capstyle="butt", zorder=2)
            ax.plot([r["dz"]], [yk], marker=mk, ms=3.0 if pooled else 3.2, mfc="white" if s == "exp1" else lc,
                    mec=lc, mew=0.7, lw=0, zorder=3)
            for xs in star_xs(r):  # asterisks drawn as markers: same look, no false text-overlap warnings
                ax.plot([xs], [yk], marker=STAR, ms=2.4, mew=0.45, color="k", lw=0, zorder=3)
    ax.axvline(0, color=st.REF_LINE, lw=st.LW_THIN, ls=(0, (3, 2)), zorder=1)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5, 0.75], ["−0.5", "", "0", "", "0.5", ""])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(r"Effect size $d_z$ (95% CI)")
    c.text(X0 + j * (AW + GAP) + AW / 2, AX_TOP - 1.2, title, ha="center", va="bottom", fontsize=st.SIZE_LABEL,
           fontweight="bold")

# row labels and task headers, in the label column (axes-fraction x, data y)
tr = blended_transform_factory(axes[0].transAxes, axes[0].transData)
xl = -1.0 / AW  # 1 mm left of the axes
for yy, (_, lab, _) in zip(ys, ROWS):
    axes[0].text(xl, yy, lab, transform=tr, ha="right", va="center", fontsize=st.SIZE_TEXT)
for task, yy in BLOCK.items():  # rotated task names, centred on their block (two lines each)
    axes[0].text(-(X0 - 5.0) / AW, float(np.mean(yy)), task.replace(" ", "\n"), transform=tr, rotation=90,
                 ha="center", va="center", fontsize=st.SIZE_LABEL, fontweight="bold", linespacing=1.0)

# sample key, on the title line of the label column
handles = [Line2D([], [], marker=mk, ms=3.2 if s != "merged" else 3.0, mfc="white" if s == "exp1" else
                  ("k" if s == "merged" else "0.45"), mec="k" if s == "merged" else "0.45",
                  color="k" if s == "merged" else "0.45", lw=1.0 if s == "merged" else st.LW_DATA, mew=0.7,
                  label=f"{lab} (N = {N[s]})") for s, lab, mk in SAMPLES]
c.fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.3 / W, 1 - 0.6 / H), ncol=3,
             handlelength=1.6, columnspacing=1.2, borderaxespad=0)

SD = SourceData("sfig1", OUT)
SD.add("", [{"task": task, "measure": key, "sample": s, "contrast": ckey,
             **{f: T2[s][key][ckey][f] for f in ("n", "dz", "dz_lo", "dz_hi", "p_holm")}}
            for task, _, key in ROWS for s, *_ in SAMPLES for ckey, *_ in CONTRASTS],
       "standardized incentive effects (dz, exact 95% CI) and Holm-corrected p per measure, sample and contrast "
       "(from paper_tables.json, table2)")
st.save(c.fig, OUT / "figures" / "sfig1")
SD.write()
print("saved sfig1", f"{W:.0f} x {H:.1f} mm", N)
