"""Supplementary Figure 2: model and parameter recovery for the single-noise models (Experiment 1).

Panels:
    a  model recovery: % of 20 simulations in which each generative model was recovered (group Bayesian model
       selection)
    b  parameter recovery for the inference-only model: generative against recovered inference noise per
       condition, and generative against recovered incentive effect (Δ log σinf)

Inputs: results/exploratory/exp1/modeling/model_identifiability/confusion_matrix_percent.csv (a);
results/exploratory/exp1/modeling/recovery_noise_per_cond/recovery_runs.csv (b).
Output: plots/exploratory/figures/sfig2.{svg,pdf,png}, at print size (180 mm wide).

Usage (from the repository root):
    python scripts/exploratory/figures/sfig2.py
"""

from __future__ import annotations

import figpaths  # noqa: F401  (paths first)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from motinf.exploratory import figure_data as R  # data preparation
from motinf.exploratory.figlib import figstyle as st
from motinf.exploratory.figlib.canvas import Canvas
from motinf.exploratory.figlib.matrix import matrix
from motinf.exploratory.figlib.source_data import SourceData

st.apply()
ROOT, OUT = figpaths.DATA_ROOT, figpaths.OUT_ROOT
EXP = "exp1"
LOSS, NEUTRAL, GAIN = st.LOSS, st.NEUTRAL, st.GAIN
SRC = ("ang", "inf", "sel")

conf = pd.read_csv(ROOT / f"results/exploratory/{EXP}/modeling/model_identifiability/confusion_matrix_percent.csv")
conf = conf.set_index("generative_model").loc[list(SRC), list(SRC)].astype(float)

df = R.load_recovery_runs(R.default_recovery_runs_csv(ROOT, EXP))
df["incentive"] = df["incentive"].astype(int)
df_ok = df.loc[df["ok"]].copy()
pt, pf = R.participant_incentive_pivots(df_ok)
eff = R.logdiff_effects_long(pt, pf)
N_RUNS, N_PART = len(df_ok), len(pt)


def r_of(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[m], y[m])[0, 1])


DOT_S = 9.0  # marker area (pt²), about 3 pt across: the size of the original dots at print scale


def dots(ax, x, y, inc):
    ax.scatter(x, y, s=DOT_S, facecolors=[st.shade(st.INC_COLORS[inc], st.DOT_SHADE)], edgecolors="none",
               linewidths=0, alpha=0.55, zorder=3)


def identity(ax, lo, hi):
    ax.plot([lo, hi], [lo, hi], color=st.REF_LINE, lw=st.LW_THIN, ls=(0, (3, 2)), zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")


def key(ax, entries, loc="upper left"):
    hs = [Line2D([], [], marker="o", ms=3.0, lw=0, mfc=st.shade(st.INC_COLORS[i], st.DOT_SHADE), mec="none",
                 label=f"{st.INC_TICKS[i]}:  r = {st.fmt_num(r)}") for i, r in entries]
    ax.legend(handles=hs, loc=loc, title=st.INC_XLABEL, alignment="left", handletextpad=0.2, borderaxespad=0.3,
              labelspacing=0.25)


W, H = st.LAYOUT_DOUBLE, 61.5  # layout mm; saved at st.DOUBLE wide (st.page_scale)
c = Canvas(W, H)
yt, y0 = 1.6, 8.6

# a  model recovery
c.letter(0, 0.5, yt)
c.title(4.5, yt + 0.2, "Model recovery")
ax = c.axes_mm(15.5, y0 + 2.0, 39.0, 39.0)
matrix(ax, conf.to_numpy(), cmap="Greys", vmin=0, vmax=100, xlabels=[st.sym("sigma", k) for k in SRC],
       xlabel="Recovered noise source", ylabel="Generative noise source", grid="0.8", grid_lw=0.4)
ax.add_patch(plt.Rectangle((-0.5, -0.5), 3, 3, fill=False, ec="0.6", lw=st.LW_AXIS, zorder=5))
ax.text(1.0, 1.02, "% of simulations", transform=ax.transAxes, ha="right", va="bottom", fontsize=st.SIZE_TEXT)

# b  parameter recovery of the inference-only model
c.letter(1, 66.0, yt)
c.title(70.0, yt + 0.2, "Parameter recovery, inference-only model")
S = 45.0
ax1 = c.axes_mm(78.5, y0, S, S)
for inc in (LOSS, NEUTRAL, GAIN):
    sub = df_ok[df_ok["incentive"] == inc]
    dots(ax1, sub["sd_inf_true"], sub["sd_inf_fit"], inc)
hi = 1.04 * max(df_ok["sd_inf_true"].max(), df_ok["sd_inf_fit"].max())
identity(ax1, 0, hi)
ax1.set_xticks([0, 1, 2, 3, 4, 5])
ax1.set_yticks([0, 1, 2, 3, 4, 5])
ax1.set_xlabel("Generative " + st.sym("sigma", "inf"))
ax1.set_ylabel("Recovered " + st.sym("sigma", "inf"))
key(ax1, [(i, r_of(*df_ok.loc[df_ok["incentive"] == i, ["sd_inf_true", "sd_inf_fit"]].to_numpy().T))
          for i in (LOSS, NEUTRAL, GAIN)])

ax2 = c.axes_mm(137.0, y0, S, S)
for inc in (LOSS, GAIN):
    sub = eff[eff["incentive"] == inc]
    dots(ax2, sub["gen"], sub["fit"], inc)
lo = 1.06 * min(eff["gen"].min(), eff["fit"].min())
hi2 = 0.6
identity(ax2, lo, hi2)
ax2.set_xticks([-2, -1, 0])
ax2.set_yticks([-2, -1, 0])
lab = r"$\Delta\log\,\sigma_{\mathrm{inf}}$ (vs £0)"
ax2.set_xlabel("Generative " + lab)
ax2.set_ylabel("Recovered " + lab)
key(ax2, [(i, r_of(*eff.loc[eff["incentive"] == i, ["gen", "fit"]].to_numpy().T)) for i in (LOSS, GAIN)])

SD = SourceData("sfig2", OUT)
SD.add("a", conf.rename_axis("generative_model").reset_index(),
       "model recovery: % of 20 simulations in which each generative model (rows) was recovered (columns)")
SD.add("b", df_ok[["participant", "incentive", "sd_inf_true", "sd_inf_fit"]], "parameter recovery of the inference-only "
       "model: generative and recovered inference noise per simulated participant and condition", part="sd_inf")
SD.add("b", eff, "generative (gen) and recovered (fit) incentive effect on log inference noise (vs neutral), "
       "participant means over runs", part="effects")
st.save(c.fig, OUT / "figures" / "sfig2")
SD.write()
print("saved sfig2", f"{W:.0f} x {H:.1f} mm", f"runs: {N_RUNS}, participants: {N_PART}",
      "| eff range", round(eff[["gen", "fit"]].min().min(), 2), round(eff[["gen", "fit"]].max().max(), 2))
