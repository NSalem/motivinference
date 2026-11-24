"""Figure 3: full model: noise at each stage and its reduction by incentives.

Panels:
    a  fitted noise of each source at £0 and ±£1 (log scale), and the incentive effects δ = log σ£0 − log σ±£1
    b  choice variance predicted by each source at £0 and ±£1, as a share of the total at £0, with the part
       removed by incentives; below, each source's share of the incentive effect
    c  reduction of choice variability under incentives against sequence length: contributions of the three
       incentive effects, and the residual up to the data
    d  error of the predicted incentive effect, for the full model and for the models without one incentive
       effect (called knockouts in the files)

Inputs (results/exploratory/merged/modeling/stats/full_model_stats/): params/{params_wide, ttests_motiv_vs0,
variance_partition_params, r2_var_red_natural_abs_obs_vs_knockout_model}.csv and knockout/{knockout_predictions,
knockout_var_red_natural_by_seqlen_long}.csv.
Output: plots/exploratory/figures/fig3.{svg,pdf,png}, at print size (180 mm wide).

Usage (from the repository root):
    python scripts/exploratory/figures/fig3.py
"""

from __future__ import annotations

import os

import figpaths  # noqa: F401  (paths first)

import numpy as np
import pandas as pd

from motinf.exploratory import figure_data as F  # Fig. 3 computations
from motinf.exploratory.figlib import dist, figstyle as st
from motinf.exploratory.figlib.canvas import Canvas
from motinf.exploratory.figlib.dist import log_axis
from motinf.exploratory.figlib.source_data import SourceData

st.apply()
ROOT, OUT = figpaths.DATA_ROOT, figpaths.OUT_ROOT
SD = SourceData("fig3", OUT)
W, H = st.LAYOUT_DOUBLE, 93.0  # layout mm; saved at st.DOUBLE wide (st.page_scale)
CH = ("ang", "inf", "sel")  # channel names in the data files
KEY = {"ang": "sen", "inf": "inf", "sel": "sel"}  # figstyle colour / symbol keys
NC, NCL = st.NOISE_COLORS, st.NOISE_COLORS_LIGHT
COL0 = {ch: NC[KEY[ch]] for ch in CH}  # £0: dark
COL1 = {ch: NCL[KEY[ch]] for ch in CH}  # ±£1: light
NAME = {"ang": "Sensory", "inf": "Inference", "sel": "Selection"}
DASH = (0, (3, 1.5))
STATS = ROOT / "results/exploratory/merged/modeling/stats/full_model_stats"

c = Canvas(W, H)
cv = c.cv

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

pars = pd.read_csv(STATS / "params/params_wide.csv")
tt = pd.read_csv(STATS / "params/ttests_motiv_vs0.csv").set_index("test")
vp = pd.read_csv(STATS / "params/variance_partition_params.csv")
ko_pred = pd.read_csv(STATS / "knockout/knockout_predictions.csv")
ko_pred["participant"] = ko_pred["participant"].astype(str)
ko_long = pd.read_csv(STATS / "knockout/knockout_var_red_natural_by_seqlen_long.csv")
mae = pd.read_csv(STATS / "params/r2_var_red_natural_abs_obs_vs_knockout_model.csv").set_index("model_id")
N_A = len(pars)
N_C = ko_long["participant"].nunique()

# --------------------------------------------------------------------------- #
# a  Noise per source and its incentive modulation
# --------------------------------------------------------------------------- #

c.letter(0, 0.5, 1.6)
c.title(4.5, 1.8, "Full model: noise at each stage and its reduction by incentives")
ya, ha = 13.0, 24.0  # axes top and height
BG_L = 10.5  # shaded background left of each axes: tick labels plus the y label, which sits ~1 mm inside
XA = {"ang": 12.5, "inf": 53.0, "sel": 93.5}  # 3 mm between backgrounds
WA = 27.0
yb0, yb1 = 6.0, 44.0  # top and bottom of the shaded backgrounds
axes_lv = {}
for ch in CH:
    x = XA[ch]
    c.box(x - BG_L, yb0, WA + BG_L, yb1 - yb0, fc=st.tint(COL0[ch], st.BG_TINT))
    c.text(x - BG_L + (WA + BG_L) / 2, 7.9, NAME[ch], ha="center", va="center", fontweight="bold",
           fontsize=st.SIZE_LABEL, zorder=4)
    ax = c.axes_mm(x, ya, WA, ha, bg=False)
    w = pars[[f"sd_{ch}", f"sd_{ch}_v"]].set_axis([0.0, 1.0], axis=1)
    dist.levels_panel(ax, w, levels=(0.0, 1.0), colors={0.0: COL0[ch], 1.0: COL1[ch]}, names=st.ABS_TICKS,
                      log=True, ylabel=st.sym("sigma", KEY[ch]), xlabel=st.ABS_XLABEL)
    axes_lv[ch] = ax
log_axis(axes_lv["ang"], [0.05, 0.1, 0.2, 0.4])
for ch in ("inf", "sel"):
    log_axis(axes_lv[ch], [0.25, 0.5, 1, 2, 4])
lims = [axes_lv[ch].get_ylim() for ch in ("inf", "sel")]  # same units (log LR): same limits
for ch in ("inf", "sel"):
    axes_lv[ch].set_ylim(min(lo for lo, _ in lims), max(hi for _, hi in lims))

XE, WE = 134.0, 38.0
c.box(XE - BG_L, yb0, WE + BG_L, yb1 - yb0, fc=st.tint(st.MUTED, st.BG_TINT))
c.text(XE - BG_L + (WE + BG_L) / 2, 7.9, "Incentive effect " + r"$\delta$", ha="center", va="center",
       fontweight="bold", fontsize=st.SIZE_LABEL, zorder=4)
ax = c.axes_mm(XE, ya, WE, ha, bg=False)
dist.effects_panel(ax, {ch: pars[f"motiv_{ch}"].to_numpy(float) for ch in CH}, contrasts=CH, colors=COL0,
                   names={ch: st.sym("delta", KEY[ch]) for ch in CH},
                   pvals={ch: float(tt.loc[f"motiv_{ch} vs 0", "p_holm"]) for ch in CH},
                   ylabel=r"$\log\sigma_0 - \log\sigma_{\pm1}$",
                   xlabel="")
dist.pct_axis(ax, "reduction", "% noise reduction")
SD.add("a", pars[[c_ for c_ in ["participant"] + [f"{p}_{ch}{v}" for ch in CH for p, v in
                                                  (("sd", ""), ("sd", "_v"), ("motiv", ""))] if c_ in pars.columns]],
       "full model per participant: noise of each source at neutral (sd_*) and with an incentive (sd_*_v), and the "
       "incentive effects (motiv_* = log sd_* − log sd_*_v)")

# --------------------------------------------------------------------------- #
# b  Variance per source, with and without incentives, and each source's share of the effect
# --------------------------------------------------------------------------- #

yt = 45.5  # letters and titles of the second row
c.letter(1, 0.5, yt)
c.title(4.5, yt + 0.2, "Predicted variance per source")
p0, p1, sem0, sem1, d_sem, _ = F.variance_partition_log_decomposition(vp, CH)
red = np.maximum(0.0, p0 - p1)  # part of the £0 variance removed by incentives, per source
slack = max(0.0, 1.0 - p1.sum())
w_red = slack * red / red.sum()
pct0 = F.integer_percents_largest_remainder(p0)
pct1 = F.integer_percents_largest_remainder(np.concatenate([p1, w_red]))
share, share_sem = F.knockout_channel_share_point_and_sem(ko_pred, CH)
pct_share = F.integer_percents_largest_remainder(share)
SD.add("b", pd.DataFrame({"source": list(CH), "share_neutral": p0, "share_neutral_sem": sem0, "share_incentive": p1,
                          "share_incentive_sem": sem1, "removed_by_incentive_drawn": w_red, "pct_neutral": pct0,
                          "pct_incentive": pct1[:3], "pct_removed": pct1[3:], "share_of_effect": share,
                          "share_of_effect_sem": share_sem, "pct_share_of_effect": pct_share}),
       "predicted choice variance per source as a share of the total at neutral, at neutral and with an incentive "
       "(mean and s.e.m. over participants), and each source's share of the incentive effect")

xb, wb = 11.0, 62.0
HB = 0.56  # bar thickness (row units)
ERR = dict(color="0.2", lw=st.LW_THIN, capsize=1.0, capthick=st.LW_THIN, zorder=5)
SMALL = 0.06  # segments narrower than this carry their value outside the bar


def stacked(ax, y, widths, labels, colors, edges, ls, sems, *, dark, outside, start=0.0):
    """One stacked bar from ``start``; values inside the segments (outside, above or below, when a segment is
    narrow). Returns the right end."""
    left = start
    for wv, lab, fc, ec, sem in zip(widths, labels, colors, edges, sems):
        if wv <= 1e-9:
            continue
        ax.barh(y, wv, left=left, height=HB, color=fc, ec=ec, lw=0.6 if ec != "white" else 0.4, ls=ls, zorder=2)
        if np.isfinite(sem) and sem > 0:
            ax.errorbar(left + wv, y + 0.3 * HB, xerr=sem, fmt="none", **ERR)
        if lab in ("0%",):
            pass  # a sliver too small to label (rounds to 0%)
        elif wv > SMALL:
            ax.text(left + wv / 2, y - 0.08 * HB, lab, ha="center", va="center", fontsize=st.SIZE_TEXT,
                    color="white" if dark else "k", zorder=6)
        else:
            off = HB / 2 + 0.05
            ax.text(left + wv / 2, y + off if outside == "above" else y - off, lab, ha="center",
                    va="bottom" if outside == "above" else "top", fontsize=st.SIZE_TEXT, zorder=6)
        left += wv
    return left


# variance per source at £0 and ±£1, as % of the £0 total; the part removed by incentives is outlined
y_bars, HB_AX = yt + 6.6, 18.0  # top and height of the axes holding the £0 and ±£1 bars
ax = c.axes_mm(xb, y_bars, wb, HB_AX)
ax.patch.set_visible(False)  # let the zoom lines (drawn on the canvas) show through
Y0, Y1 = 1.0, 0.0
stacked(ax, Y0, p0, [f"{v}%" for v in pct0], [COL0[ch] for ch in CH], ["white"] * 3, "-", sem0, dark=True,
        outside="above")
end1 = stacked(ax, Y1, p1, [f"{v}%" for v in pct1[:3]], [COL1[ch] for ch in CH], ["white"] * 3, "-", sem1,
               dark=False, outside="below")
x_red0 = end1
x_red1 = stacked(ax, Y1, w_red, [f"{v}%" for v in pct1[3:]], ["white"] * 3, [COL0[ch] for ch in CH], DASH,
                 [np.nan] * 3, dark=False, outside="below", start=end1)
ax.text(1.015, Y1, f"−{sum(pct1[3:])}%", ha="left", va="center", fontsize=st.SIZE_TEXT, fontweight="bold")
ax.set_xlim(0, 1)
ax.set_ylim(-0.75, 1.62)
ax.set_yticks([Y0, Y1], [st.ABS_TICKS[0.0], st.ABS_TICKS[1.0]])
ax.set_ylabel(st.ABS_XLABEL, labelpad=1)
ax.set_title("% of the total at £0", fontsize=st.SIZE_TEXT, loc="right", pad=1.0)
for side in ("left", "bottom"):
    ax.spines[side].set_visible(False)
ax.set_xticks([])
ax.tick_params(axis="y", length=0)

# each source's share of the incentive effect (knockout contrasts), zoomed out of the outlined part
yz, hz = yt + 31.6, 5.5
xz0, wz = xb + 14.0, wb - 14.0
axz = c.axes_mm(xz0, yz, wz, hz)
axz.patch.set_visible(False)
stacked(axz, 0.0, share, [f"{v}%" for v in pct_share], [COL1[ch] for ch in CH], [COL0[ch] for ch in CH], DASH,
        share_sem, dark=False, outside="below")
axz.set_xlim(0, 1)
axz.set_ylim(-HB / 2 - 0.02, HB / 2 + 0.02)
axz.set_xticks([])
axz.set_yticks([0], ["Share of the\nincentive effect"])
axz.tick_params(axis="y", length=0)
for side in ("left", "bottom"):
    axz.spines[side].set_visible(False)
# zoom lines: from the outlined part of the ±£1 bar to the ends of the share bar
y_bar_bottom = y_bars + HB_AX * (1.62 - (Y1 - HB / 2)) / (1.62 + 0.75)
for xf, xt in ((x_red0, 0.0), (x_red1, 1.0)):
    cv.plot([xb + wb * xf, xz0 + wz * xt], [y_bar_bottom, yz], color="0.6", lw=st.LW_THIN, zorder=1)

# --------------------------------------------------------------------------- #
# c  Incentive effect over sequence length, per modulation
# --------------------------------------------------------------------------- #

xc = 84.0
c.letter(2, xc, yt)
c.title(xc + 4, yt + 0.2, "Incentive effect per source")
eff = F.motiv_effect_per_channel_long(ko_long)
m = eff.groupby("seqlen")[list(CH)].mean().sort_index()
xs = m.index.to_numpy(float)
cumc = m.cumsum(axis=1)
dd = ko_long[ko_long["model_id"] == "data"].groupby("seqlen")["var_red"]
dm, dse = dd.mean().reindex(m.index), dd.sem().reindex(m.index)
SD.add("c", pd.DataFrame({"seqlen": xs, **{f"effect_of_motiv_{ch}_mean": m[ch].to_numpy(float) for ch in CH},
                          "data_reduction_mean": dm.to_numpy(float), "data_reduction_sem": dse.to_numpy(float)}),
       f"reduction of choice variability (neutral − incentive) per sequence length: contribution of each incentive "
       f"effect (stacked in the figure) and data, mean and s.e.m. over participants (N = {N_C})", part="summary")
SD.add("c", ko_long, "per participant and sequence length: observed and predicted reductions (model_id)",
       part="participants")
ax = c.axes_mm(xc + 9.0, yt + 4.75, 34.0, 34.0)
lower = np.zeros_like(xs)
for ch in CH:
    upper = cumc[ch].to_numpy(float)
    ax.fill_between(xs, lower, upper, color=COL1[ch], lw=0, zorder=1)
    ax.plot(xs, upper, color=COL0[ch], lw=st.LW_THIN, zorder=2)
    lower = upper
ax.fill_between(xs, lower, dm.to_numpy(float), color="0.88", lw=0, zorder=1)
ax.errorbar(xs, dm, yerr=dse, fmt="o", ms=2.4, color="k", mec="none", elinewidth=st.LW_AXIS, zorder=4)
ax.axhline(0, color="k", lw=st.LW_AXIS, zorder=0)
# direct labels at the right end, at the middle of each band, pushed apart where bands are thin
edges = [0.0] + [cumc[ch].iloc[-1] for ch in CH] + [dm.iloc[-1]]
labs = [st.sym("delta", KEY[ch]) for ch in CH] + ["residual"]
ys_ = [(edges[i] + edges[i + 1]) / 2 for i in range(4)]
GAP = 0.075  # minimum spacing (data units, about 2.5 mm)
ys_[0] = max(ys_[0], 0.035)
for i in range(1, 4):
    ys_[i] = max(ys_[i], ys_[i - 1] + GAP)
for yv, lab in zip(ys_, labs):
    ax.text(12.6, yv, lab, ha="left", va="center", fontsize=st.SIZE_TEXT)
ax.text(11.35, dm.iloc[-1], "data", ha="right", va="center", fontsize=st.SIZE_TEXT)
ax.set_xlim(3.6, 12.4)
ax.set_xticks([4, 8, 12])
ax.spines["bottom"].set_bounds(4, 12)
ax.set_ylim(0, None)
ax.set_xlabel(st.SEQ_LENGTH)
ax.set_ylabel("Reduction in choice variability\n(£0 − ±£1)", linespacing=1.0)

# --------------------------------------------------------------------------- #
# d  Model lesioning: error of the predicted incentive effect
# --------------------------------------------------------------------------- #

xd = 138.0
c.letter(3, xd, yt)
c.title(xd + 4, yt + 0.2, "Model lesioning")
models = [("full", "Full model", "k"), ("ko_ang", "no " + st.sym("delta", "sen"), COL0["ang"]),
          ("ko_inf", "no " + st.sym("delta", "inf"), COL0["inf"]), ("ko_sel", "no " + st.sym("delta", "sel"), COL0["sel"])]
ax = c.axes_mm(xd + 10.3, yt + 3.7, 29.0, 34.0)
for j, (mid, _, col) in enumerate(models):
    yv = len(models) - 1 - j
    ax.errorbar(mae.loc[mid, "mean_mae"], yv, xerr=mae.loc[mid, "sem_mae"], fmt="o", ms=3.0, color=col, mec="none",
                elinewidth=st.LW_DATA, zorder=3)
ax.axvline(mae.loc["full", "mean_mae"], color=st.REF_LINE, lw=st.LW_THIN, ls=(0, (3, 2)), zorder=1)
ax.set_yticks(range(len(models)), [lab for _, lab, _ in models][::-1])
ax.set_ylim(-0.6, len(models) - 0.4)
ax.set_xlim(0.95, 1.65)
ax.set_xticks([1.0, 1.2, 1.4, 1.6], ["1.0", "1.2", "1.4", "1.6"])
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.set_xlabel("Error of the predicted\nincentive effect (MAE)", linespacing=1.0)

if os.environ.get("FIG_DEBUG"):  # horizontal extent of each axes and of its labels (mm), for layout
    c.fig.canvas.draw()
    rr = c.fig.canvas.get_renderer()
    k_ = 25.4 / c.fig.dpi
    for a in c.fig.axes[1:]:
        p_, t_ = a.get_window_extent(rr), a.get_tightbbox(rr)
        print(f"axes {p_.x0 * k_:6.1f}-{p_.x1 * k_:6.1f} | labels {t_.x0 * k_:6.1f}-{t_.x1 * k_:6.1f} "
              f"| y {H - p_.y1 * k_:5.1f}-{H - p_.y0 * k_:5.1f} | labels y {H - t_.y1 * k_:5.1f}-{H - t_.y0 * k_:5.1f}")
SD.add("d", mae.loc[[mid for mid, _, _ in models]].reset_index(),
       "mean absolute error of the predicted incentive effect, full model and models without one incentive effect "
       "(knockouts): mean and s.e.m. over participants")
st.save(c.fig, OUT / "figures" / "fig3")
SD.write()
print("saved", OUT / "figures" / "fig3.svg", f"| N a: {N_A}, N c–d: {N_C}",
      "| b:", pct0, pct1, pct_share)
