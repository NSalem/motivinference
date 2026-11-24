"""Supplementary Figure 5: parameter recovery for the full model.

Pearson correlation between generative (rows) and recovered (columns) parameters over 216 simulated datasets (one
per participant of Experiments 1 and 2), ± s.e.m. over 2,000 bootstrap resamples. Outlined: baseline noise σ and
incentive effects δ.

Input: results/exploratory/merged/modeling/param_recovery_full_model/parameter_recovery_runs.csv (fits with
ok = False are left out).
Output: plots/exploratory/figures/sfig5.{svg,pdf,png}, at print size (89 mm wide).

Usage (from the repository root):
    python scripts/exploratory/figures/sfig5.py
"""

from __future__ import annotations

import figpaths  # noqa: F401  (paths first)

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory

from motinf.exploratory import figure_data as P  # computation
from motinf.exploratory.figlib import figstyle as st
from motinf.exploratory.figlib.canvas import Canvas
from motinf.exploratory.figlib.matrix import matrix
from motinf.exploratory.figlib.source_data import SourceData

st.apply()
ROOT, OUT = figpaths.DATA_ROOT, figpaths.OUT_ROOT
runs = pd.read_csv(ROOT / "results/exploratory/merged/modeling/param_recovery_full_model/parameter_recovery_runs.csv")
runs = runs[runs["ok"].astype(bool)].copy()
gen = np.column_stack([pd.to_numeric(runs[f"{p}_true"]).to_numpy(float) for p in P.PARNAMES])
rec = np.column_stack([pd.to_numeric(runs[P.RECOVERED_COLUMN.get(p, f"{p}_fit")]).to_numpy(float) for p in P.PARNAMES])
n = len(P.PARNAMES)
R = np.full((n, n), np.nan)
SE = np.full((n, n), np.nan)
for i in range(n):  # same loop, seeds and bootstrap as the original
    for j in range(n):
        R[i, j], SE[i, j] = P.bootstrap_corr_sem(rec[:, j], gen[:, i], n_boot=2000, seed=123 + 17 * i + j)

labels = [st.sym("sigma", k) for k in ("ang", "inf", "sel")] + [st.sym("delta", k) for k in ("ang", "inf", "sel")]
W, H = st.SINGLE, 74.0
c = Canvas(W, H)
X0, Y0, S = 12.0, 6.0, 60.0
ax = c.axes_mm(X0, Y0, S, S)
im = matrix(ax, R, cmap=sns.color_palette("vlag", as_cmap=True), vmin=-1, vmax=1, xlabels=labels,
            fmt=lambda i, j, v: f"{st.fmt_num(v)}\n±{SE[i, j]:.2f}", grid="white", grid_lw=0.5)
for k in (0, 3):
    ax.add_patch(Rectangle((k - 0.5, k - 0.5), 3, 3, fill=False, ec="0.25", lw=0.8, zorder=5, clip_on=False))
ax.set_ylabel("Generative")
ax.set_xlabel("Recovered")
ax.set_xlim(-0.5, n - 0.5)
ax.set_ylim(n - 0.5, -0.5)
# column groups above the matrix
tr = blended_transform_factory(ax.transData, ax.transAxes)
for k, lab in ((1, "Baseline noise"), (4, "Incentive effect (log)")):
    ax.plot([k - 1.4, k + 1.4], [1.025, 1.025], transform=tr, color="0.25", lw=st.LW_THIN, clip_on=False)
    ax.text(k, 1.04, lab, transform=tr, ha="center", va="bottom", fontsize=st.SIZE_TEXT)
# colour bar
cax = c.axes_mm(X0 + S + 3.0, Y0 + 6.0, 2.2, S - 12.0)
cb = c.fig.colorbar(im, cax=cax, ticks=[-1, -0.5, 0, 0.5, 1])
cb.ax.set_yticklabels(["−1", "−0.5", "0", "0.5", "1"])
cb.outline.set_linewidth(st.LW_AXIS)
cb.ax.tick_params(width=st.LW_AXIS, length=1.5)
cb.set_label("Pearson r", labelpad=1.5)

SD = SourceData("sfig5", OUT)
SD.add("", [{"generative": P.PARNAMES[i], "recovered": P.PARNAMES[j], "pearson_r": R[i, j], "sem_bootstrap": SE[i, j]}
            for i in range(n) for j in range(n)],
       f"Pearson correlation between generative and recovered parameters over {len(runs)} simulated datasets, with "
       "its s.e.m. over 2,000 bootstrap resamples", part="matrix")
SD.add("", runs[["participant"] + [f"{p}_true" for p in P.PARNAMES]
                + [P.RECOVERED_COLUMN.get(p, f"{p}_fit") for p in P.PARNAMES]],
       "generative and recovered parameters of each simulated dataset", part="runs")
st.save(c.fig, OUT / "figures" / "sfig5")
SD.write()
print("saved sfig5", f"{W:.0f} x {H:.0f} mm", "runs:", len(runs))
print(np.round(R, 2))
