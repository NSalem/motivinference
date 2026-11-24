"""Supplementary Figure 4: single-modulation models (incentives modulate one source of noise).

Panels:
    a  Bayesian model selection, both experiments pooled (N = 216)
    b  model recovery: % of 20 simulations of Experiment 1 in which each modulated source of noise was recovered
    c  error of each model's predicted incentive effect (mean absolute error per participant; N = 163)
    d  reduction of choice variability under incentives against sequence length: data and each model's prediction

Inputs: results/exploratory/merged/modeling/stats/model_fit_stats/params/{groupbmc_incentive_effects,
r2_var_red_natural_abs_obs_vs_inc_eff_model, var_red_by_seqlen_single_modulation}.csv (a, c, d);
results/exploratory/exp1/modeling/model_motiv_identifiability/confusion_matrix_percent.csv (b).
Output: plots/exploratory/figures/sfig4.{svg,pdf,png}, at print size (180 mm wide).

Usage (from the repository root):
    python scripts/exploratory/figures/sfig4.py
"""

from __future__ import annotations

import figpaths  # noqa: F401  (paths first)

import pandas as pd
from matplotlib.patches import Rectangle

from motinf.exploratory.figlib import figstyle as st
from motinf.exploratory.figlib.canvas import Canvas
from motinf.exploratory.figlib.matrix import matrix
from motinf.exploratory.figlib.source_data import SourceData

st.apply()
ROOT, OUT = figpaths.DATA_ROOT, figpaths.OUT_ROOT
NC = st.NOISE_COLORS
MODELS = {"ang_v": "sen", "inf_v": "inf", "sel_v": "sel"}  # model -> modulated noise source
SRC_NAME = {"sen": "Sensory noise", "inf": "Inference noise", "sel": "Selection noise"}
PARAMS = ROOT / "results/exploratory/merged/modeling/stats/model_fit_stats/params"

bms = pd.read_csv(PARAMS / "groupbmc_incentive_effects.csv").set_index("model")
N_BMS = int(bms["n"].iloc[0])
conf = pd.read_csv(ROOT / "results/exploratory/exp1/modeling/model_motiv_identifiability/confusion_matrix_percent.csv")
order = ["ang_motiv", "inf_motiv", "sel_motiv"]
conf = conf.set_index("generative_model").loc[order, order].astype(float)
mae = pd.read_csv(PARAMS / "r2_var_red_natural_abs_obs_vs_inc_eff_model.csv").set_index("model_id")
N_MAE = int(mae["n_participants"].iloc[0])

# reduction of choice variability per sequence length: data and each model's prediction (run_inference_model_stats.py)
red = pd.read_csv(PARAMS / "var_red_by_seqlen_single_modulation.csv", float_precision="round_trip")
red = red[red["incentive"] == 1.0]

W = st.LAYOUT_DOUBLE  # layout mm; saved at st.DOUBLE wide (st.page_scale)
c = Canvas(W, 101.0)
cv = c.cv
yt, y0 = 1.6, 10.0

# a  Bayesian model selection (pooled)
c.letter(0, 0.5, yt)
c.title(4.5, yt + 0.2, "Model comparison")
ax = c.axes_mm(15.0, y0, 6.0, 33.0)
bottom = 0.0
seg = {}
for m in ("inf_v", "sel_v", "ang_v"):
    f = float(bms.loc[m, "frequency_mean"])
    ax.bar(0, f, bottom=bottom, width=1.0, color=NC[MODELS[m]], ec="white", lw=0.3, zorder=2)
    seg[m] = (bottom, f)
    bottom += f
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(0, 1)
ax.set_xticks([])
ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
ax.spines["bottom"].set_visible(False)
ax.set_ylabel(f"Model frequency (N = {N_BMS})")
assert bms.loc["inf_v", "pxp"] > 0.99
xl = 22.6  # direct labels, right of the bar
y_of = lambda v: y0 + 33.0 * (1 - v)  # noqa: E731
f_inf, f_sel, f_ang = (float(bms.loc[m, "frequency_mean"]) for m in ("inf_v", "sel_v", "ang_v"))
c.text(xl, y_of(f_inf / 2) - 1.3, f"{st.sym('sigma', 'inf')} modulated: {100 * f_inf:.0f}%", va="center")
c.text(xl, y_of(f_inf / 2) + 1.3, "PXP > .99", va="center")
c.text(xl, y_of(1.0) + 0.6, f"{st.sym('sigma', 'sel')}: {100 * f_sel:.0f}%,  {st.sym('sigma', 'ang')}: "
       + ("< 1%" if f_ang < 0.01 else f"{100 * f_ang:.0f}%"), va="center")

# b  model recovery
c.letter(1, 55.0, yt)
c.title(59.0, yt + 0.2, "Model recovery")
ax = c.axes_mm(70.0, y0 + 1.0, 32.0, 32.0)
matrix(ax, conf.to_numpy(), cmap="Greys", vmin=0, vmax=100, xlabels=[st.sym("sigma", k) for k in ("ang", "inf", "sel")],
       xlabel="Recovered modulated source", ylabel="Generative modulated source", grid="0.8", grid_lw=0.4)
ax.add_patch(Rectangle((-0.5, -0.5), 3, 3, fill=False, ec="0.6", lw=st.LW_AXIS, zorder=5))
ax.text(1.0, 1.02, "% of simulations", transform=ax.transAxes, ha="right", va="bottom", fontsize=st.SIZE_TEXT)

# c  error of the predicted incentive effect
c.letter(2, 118.0, yt)
c.title(122.0, yt + 0.2, "Fit to the incentive effect")
ax = c.axes_mm(136.0, y0 + 1.0, 44.0, 26.0)
for k, m in enumerate(MODELS):
    yv = 2 - k
    ax.errorbar(mae.loc[m, "mean_mae"], yv, xerr=mae.loc[m, "sem_mae"], fmt="o", ms=3.0, color=NC[MODELS[m]],
                mec="none", elinewidth=st.LW_DATA * 1.3, zorder=3)
ax.axvline(mae.loc["inf_v", "mean_mae"], color=st.REF_LINE, lw=st.LW_THIN, ls=(0, (3, 2)), zorder=1)
ax.set_yticks([2, 1, 0], [st.sym("sigma", MODELS[m][:3] if MODELS[m] != "sen" else "ang") + " modulated"
                          for m in MODELS])
ax.set_ylim(-0.6, 2.6)
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.set_xlim(1.0, 1.7)
ax.set_xticks([1.0, 1.2, 1.4, 1.6])
ax.set_xlabel(f"Error of the predicted\nincentive effect (MAE, N = {N_MAE})", linespacing=1.0)

# d  predictions over sequence length
yd = 52.0
c.letter(3, 0.5, yd)
c.title(4.5, yd + 0.2, "Incentive effect predicted when incentives modulate one source of noise")
xk = 134.0  # key on the title line
cv.plot([xk], [yd + 1.6], "o", ms=3.0, color="k")
c.text(xk + 1.6, yd + 1.6, "data", va="center")
cv.add_patch(Rectangle((xk + 10.5, yd + 0.8), 5.0, 1.6, fc="0.85", ec="none"))
cv.plot([xk + 10.5, xk + 15.5], [yd + 1.6, yd + 1.6], color="0.3", lw=st.LW_DATA * 1.3)
c.text(xk + 16.5, yd + 1.6, "model (mean ± s.e.m.)", va="center")
BOX = {"sen": (5.0, 59.4), "inf": (65.0, 120.9), "sel": (127.0, 181.6)}  # as Fig. 2a
yb0, yb1 = yd + 4.4, 100.3
ylim = (-0.08, 1.05)
for k, (m, src) in enumerate(MODELS.items()):
    x0, x1 = BOX[src]
    c.box(x0, yb0, x1 - x0, yb1 - yb0, fc=st.tint(NC[src], st.BG_TINT))
    c.text((x0 + x1) / 2, yb0 + 2.2, f"{SRC_NAME[src]}  {st.sym('sigma', src)}  modulated", ha="center",
           va="center", fontweight="bold", fontsize=st.SIZE_LABEL, zorder=4)
    ax = c.axes_mm(x0 + 15.0, yb0 + 6.0, 34.0, 26.0, bg=False)
    d = red[red["model"] == m].sort_values("seqlen")
    L = d["seqlen"].to_numpy(float)
    ax.fill_between(L, d["pred_red_mean"] - d["pred_red_sem"], d["pred_red_mean"] + d["pred_red_sem"],
                    color=st.tint(NC[src], 0.35), lw=0, zorder=2)
    ax.plot(L, d["pred_red_mean"], color=NC[src], lw=st.LW_DATA * 1.3, zorder=3)
    ax.errorbar(L, d["beh_red_mean"], yerr=d["beh_red_sem"], fmt="o", ms=3.0, color="k", mec="none",
                elinewidth=st.LW_DATA, zorder=4)
    ax.axhline(0, color=st.REF_LINE, lw=st.LW_THIN, ls=(0, (3, 2)), zorder=1)
    ax.set_xlim(3.2, 12.8)
    ax.set_xticks([4, 8, 12])
    ax.set_ylim(*ylim)
    ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_xlabel(st.SEQ_LENGTH)
    ax.set_ylabel("Reduction in choice\nvariability (£0 − ±£1)", linespacing=1.0)

SD = SourceData("sfig4", OUT)
SD.add("a", bms.loc[list(MODELS)].reset_index(), f"group Bayesian model selection among the single-modulation models, "
       f"both experiments pooled (N = {N_BMS}): estimated frequencies (frequency_mean, drawn) and PXP")
SD.add("b", conf.rename_axis("generative_model").reset_index(),
       "model recovery: % of 20 simulations in which each generative model (rows) was recovered (columns)")
SD.add("c", mae.loc[list(MODELS)].reset_index(), f"mean absolute error of each model's predicted incentive effect: mean "
       f"and s.e.m. over participants (N = {N_MAE})")
SD.add("d", red, "reduction of choice variability (neutral − incentive) per sequence length: data and each model's "
       "prediction, mean and s.e.m. over participants")
st.save(c.fig, OUT / "figures" / "sfig4")
SD.write()
print("saved sfig4", f"{W:.0f} x 101 mm", f"N bms {N_BMS}, N mae {N_MAE}")
print(red[["model", "seqlen", "beh_red_mean", "pred_red_mean"]].round(3).to_string(index=False))
print(mae[["mean_mae", "sem_mae"]].round(3))
