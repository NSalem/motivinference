"""Figure 2: noisy-inference model and incentive effect on inference noise.

Panels:
    a  the model: one column per source of noise (sensory, inference, selection), with what it corrupts, its
       equation, and the predicted effect of an incentive that reduces it (simulated)
    b  Bayesian model selection among the three single-noise models, per condition (N = 216)
    c  inference noise of the inference-only model per condition and its change from £0 (log scale)
    d  choice variability against sequence length, data and inference-only model, and its reduction under
       incentives (N = 163)

Inputs: results/exploratory/merged/modeling/stats/model_fit_stats/params/groupbmc_noise_per_incentive.csv (b);
results/prereg/<exp>/model_fit/stats/sd_inf_long.csv (c); results/prereg/<exp>/behavior/data_mean_inc_seqlen.csv
and results/exploratory/merged/behavior/stats/behav_extra/params/model_pred_per_seqlen_simple_inf.csv (d);
results/exploratory/merged/paper_tables/paper_tables.json (stars of c: Table 2).
Output: plots/exploratory/figures/fig2.{svg,pdf,png}, at print size (180 mm wide).

Usage (from the repository root):
    python scripts/exploratory/figures/fig2.py
"""

from __future__ import annotations

import os

import figpaths  # noqa: F401  (paths first)

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Polygon
from scipy.stats import norm

from motinf.exploratory import choice_var_seqlen_preprocess as CV
from motinf.exploratory import figure_data as FD  # Fig. 2d data preparation
from motinf.exploratory.figlib import dist, figstyle as st
from motinf.exploratory.figlib.canvas import Canvas, gabor
from motinf.exploratory.figlib.dist import log_axis
from motinf.exploratory.behav_stats_paths import behav_extra_dir
from motinf.exploratory.cohorts import cohort_subexperiments, merge_behavior_seqlen, merge_sd_inf_long
from motinf.exploratory.figlib.source_data import SourceData

st.apply()
ROOT, OUT = figpaths.DATA_ROOT, figpaths.OUT_ROOT
SD = SourceData("fig2", OUT)
SD.note("a", "model schematic with illustrative simulated values (no data)")
W, H = st.LAYOUT_DOUBLE, 108.5  # layout mm; saved at st.DOUBLE wide (st.page_scale)
LOSS, NEUTRAL, GAIN = st.LOSS, st.NEUTRAL, st.GAIN
PINK, BLUE = "#e377c2", "#4a90d9"  # category colours of the Integration task (as in Fig. 1)
NC, NCL = st.NOISE_COLORS, st.NOISE_COLORS_LIGHT
SRC = ("sen", "inf", "sel")
SRC_NAME = {"sen": "Sensory noise", "inf": "Inference noise", "sel": "Selection noise"}
DELTA = 0.29  # illustrative noise reduction in a (log units): the size of the observed effect on σ_inf
F_INC = np.exp(-DELTA)  # noise scale under incentives (σ × e^−δ)
DASH = (0, (3, 1.5))  # ±£1 (less noise) throughout panel a; £0 is solid

c = Canvas(W, H)
cv = c.cv

# --------------------------------------------------------------------------- #
# a  Model: one column per noise source
# --------------------------------------------------------------------------- #

c.letter(0, 0.5, 1.6)
c.title(4.5, 1.8, "Noisy inference model: three sources of noise")
xk, yk = 78.27, 2.98  # line-style key for all of panel a, on the title line, above the inference column
for j, (lab, ls) in enumerate((("£0", "-"), ("±£1", DASH))):
    xx_ = xk + 16.0 * j
    cv.plot([xx_, xx_ + 5.0], [yk, yk], color="0.25", lw=st.LW_DATA, ls=ls)
    c.text(xx_ + 6.0, yk, lab, va="center")
COL = {"sen": (5.0, 61.0), "inf": (65.0, 123.0), "sel": (127.0, 183.0)}  # content of each column
BOX = {"sen": (5.0, 59.4), "inf": (65.0, 120.9), "sel": (127.0, 181.6)}  # shaded backgrounds: left, right
yb0, yb1 = 5.8, 67.3  # top and bottom of the backgrounds
for k in SRC:
    x0, x1 = BOX[k]
    c.box(x0, yb0, x1 - x0, yb1 - yb0, fc=st.tint(NC[k], st.BG_TINT))
    c.text((COL[k][0] + COL[k][1]) / 2, 8.0, f"{SRC_NAME[k]}  {st.sym('sigma', k)}", ha="center", va="center",
           fontweight="bold", fontsize=st.SIZE_LABEL, zorder=4)
y_arr, l_arr = 35.5, 3.1  # processing order: arrows centred in the gaps between the backgrounds
for a_, b_ in (("sen", "inf"), ("inf", "sel")):
    xm = (BOX[a_][1] + BOX[b_][0]) / 2
    c.arrow(xm - l_arr / 2, y_arr, xm + l_arr / 2, y_arr, lw=0.8)

# -- sensory: θ (grating) → θ̂ (perceived orientation) → ℓ(θ̂) (evidence) --------------------------------
x0 = COL["sen"][0]
xs = [x0 + 14.0, x0 + 26.5, x0 + 45.0]  # items 1, 2 and n; "…" between 2 and n
xdots = x0 + 35.75
Y0 = 16.5  # top of the pictures and graphs (the equations sit above them, under the header strips)
y_g, y_p, y_e = Y0 + 4.0, Y0 + 14.7, Y0 + 23.7  # rows: grating, perceived orientation, evidence baseline
dg = 9.0
theta = [35.0, -20.0, 60.0]
theta_hat = [28.0, -31.0, 52.0]
SIG_ILLUS = 14.0  # half-width of the drawn spread (deg), illustrative
for lab, y in ((r"$\theta_i$", y_g), (r"$\hat{\theta}_i$", y_p), (r"$\ell(\hat{\theta}_i)$", y_e - 1.2)):
    c.text(x0 + 0.6, y, lab, va="center", fontsize=st.SIZE_LABEL)


def bowtie(cx, cy, th, half, r, **kw):
    """Axial wedge (both directions) of half-width ``half`` deg around orientation th (clockwise from vertical)."""
    for base in (0.0, 180.0):
        a = np.deg2rad(np.linspace(th - half, th + half, 30) + base)
        pts = np.column_stack([cx + r * np.sin(a), cy - r * np.cos(a)])
        cv.add_patch(Polygon(np.vstack([[cx, cy], pts]), closed=True, **kw))


for i, (x, th, thh) in enumerate(zip(xs, theta, theta_hat)):
    c.image(gabor(th, seed=40 + i, n=111), x, y_g, dg, z=3)
    cv.add_patch(Circle((x, y_p), dg / 2, fc="white", ec="0.6", lw=st.LW_THIN, zorder=3))
    bowtie(x, y_p, th, 2 * SIG_ILLUS, dg / 2, fc=NCL["sen"], ec="none", alpha=0.45, zorder=3.1)
    for sgn in (-1, 1):  # edges of the narrower spread under incentives
        c.orient_line(x, y_p, th + sgn * 2 * SIG_ILLUS * F_INC, dg / 2, color=NC["sen"], lw=0.5, ls=DASH, zorder=3.2)
    c.orient_line(x, y_p, th, dg / 2, color="k", lw=st.LW_THIN, ls=(0, (2, 1.2)), zorder=3.3)
    c.orient_line(x, y_p, thh, dg / 2, color=NC["sen"], lw=1.0, solid_capstyle="round", zorder=3.4)
    ell = np.sin(np.deg2rad(2 * thh))  # log LR (blue : pink), amplitude 1 for κ = 0.5 (Fig. 1c)
    hb = 3.4 * ell
    c.rbox(x - 1.1, y_e - max(hb, 0), 2.2, abs(hb), fc=BLUE if ell > 0 else PINK, r=0.15, z=3)
for y in (y_g, y_p, y_e - 1.6):
    c.text(xdots, y, "…", ha="center", va="center", fontsize=st.SIZE_LABEL)
cv.plot([xs[0] - 4.0, xs[-1] + 4.0], [y_e, y_e], color=st.REF_LINE, lw=st.LW_THIN, zorder=2.9)

# -- inference: accumulated evidence over samples, without and with inference noise ---------------------
ell_seq = np.array([np.sin(np.deg2rad(2 * theta_hat[0])), np.sin(np.deg2rad(2 * theta_hat[1])),
                    0.45, 0.70, -0.30, 0.55, 0.35, np.sin(np.deg2rad(2 * theta_hat[2]))])
n = ell_seq.size
L = np.cumsum(ell_seq)
rng = np.random.default_rng(3)
s_inf = 0.55  # illustrative
N_PATHS = 8
z = np.concatenate([np.zeros((N_PATHS, 1)), np.cumsum(rng.normal(0, 1.0, (N_PATHS, n - 1)), axis=1)], axis=1)
paths = L + s_inf * z  # £0
paths_inc = L + s_inf * F_INC * z  # ±£1: the same noise draws, scaled by e^−δ, so each stays closer to the noise-free path
ax = c.axes_mm(78.0, Y0, 38.0, 21.0, bg=False)
samp = np.arange(1, n + 1)
for p in paths:
    ax.plot(samp, p, color=NC["inf"], lw=0.45, alpha=0.55, zorder=2)
for p in paths_inc:  # dash scaled to the thinner line, so it reads like DASH at 0.75 pt
    ax.plot(samp, p, color=NC["inf"], lw=0.45, alpha=0.75, ls=(0, (5, 2.5)), zorder=2.05)
ax.plot(samp, L, color="k", lw=st.LW_DATA, marker="o", ms=1.8, zorder=3)
# distribution of the final evidence across repetitions of the same sequence
sd_n = s_inf * np.sqrt(n - 1)
yy = np.linspace(L[-1] - 3 * sd_n, L[-1] + 3 * sd_n, 200)
dens = norm.pdf(yy, L[-1], sd_n)
dens = 1.1 * dens / dens.max()
xv = n + 0.35
ax.fill_betweenx(yy, xv, xv + dens, color=NC["inf"], alpha=0.25, lw=0, zorder=2)
ax.plot(xv + dens, yy, color=NC["inf"], lw=st.LW_THIN, zorder=2.1)
dens_inc = norm.pdf(yy, L[-1], sd_n * F_INC) * 1.1 / norm.pdf(L[-1], L[-1], sd_n)  # same area, narrower
ax.plot(xv + dens_inc, yy, color=NC["inf"], lw=st.LW_DATA, ls=DASH, zorder=2.2)
ax.axhline(0, color=st.REF_LINE, lw=st.LW_THIN, ls=(0, (3, 2)), zorder=1)
ax.set_xlim(0.6, n + 1.95)
ax.set_xticks([1, 2, n], ["1", "2", "n"])
ax.set_xlabel("Sample", labelpad=1)
ax.set_ylim(min(paths.min(), yy.min()) - 0.2, max(paths.max(), yy.max()) + 0.2)
ax.set_yticks([0])
ax.set_ylabel("Σ log LR\n(A : B)", labelpad=1, linespacing=1.0)
ax.spines["bottom"].set_bounds(1, n)

# -- selection: choice = sign of the final evidence, corrupted once ------------------------------------
ax = c.axes_mm(141.0, Y0, 34.0, 21.0, bg=False)
xx = np.linspace(-3, 3, 301)
ax.plot([-3, 0, 0, 3], [0, 0, 1, 1], color="k", lw=st.LW_DATA, zorder=3)
ax.plot(xx, norm.cdf(xx / 1.0), color=NC["sel"], lw=1.0, zorder=2)
ax.plot(xx, norm.cdf(xx / F_INC), color=NC["sel"], lw=1.0, ls=DASH, zorder=2.1)
ax.axhline(0.5, color=st.REF_LINE, lw=st.LW_THIN, ls=(0, (3, 2)), zorder=1)
ax.text(-0.25, 0.97, r"$\sigma_{\mathrm{sel}} = 0$", ha="right", va="top", fontsize=st.SIZE_TEXT)
ax.text(1.2, 0.78, r"$\sigma_{\mathrm{sel}} > 0$", ha="left", va="top", fontsize=st.SIZE_TEXT)
ax.set_xlim(-3, 3)
ax.set_ylim(-0.02, 1.02)
ax.set_xticks([-2, 0, 2], ["−2", "0", "2"])
ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
ax.set_xlabel(r"Final evidence $L_n$", labelpad=1)
ax.set_ylabel("P(A)", labelpad=1)

# -- equations ------------------------------------------------------------------------------------------
y_eq = 13.9  # baseline, just under the header strips
EQ = {"sen": r"$\hat{\theta}_i = \theta_i + \varepsilon$,   $\varepsilon$ ~ $N(0, \sigma_{\mathrm{%s}})$"
             % st.SENSORY_SUBSCRIPT,
      "inf": r"$L_i = L_{i-1} + \ell(\hat{\theta}_i) + \varepsilon$  ($i \geq 2$),   $\varepsilon$ ~ $N(0, \sigma_{\mathrm{inf}})$",
      "sel": r"A if $L_n + \varepsilon > 0$,   $\varepsilon$ ~ $N(0, \sigma_{\mathrm{sel}})$"}
for k in SRC:
    x0, x1 = COL[k]
    c.text((x0 + x1) / 2, y_eq, EQ[k], ha="center", va="baseline", fontsize=st.SIZE_TEXT)

# -- predictions: an incentive that reduces this noise (σ × e^−δ) -----------------------------------------
yp, hp = 48.0, 12.5
c.text(3.85, yp + hp / 2, "Predictions", rotation=90, ha="center", va="center", fontsize=st.SIZE_LABEL,
       fontweight="bold")
nn = np.linspace(4, 12, 50)
f = F_INC
cvar = {"sen": lambda m: 0.5 * m, "inf": lambda m: 0.55 * (m - 1), "sel": lambda m: 4.0 + 0 * m}
for k in SRC:
    x0, _ = COL[k]
    # Estimation: error SD at £0 and ±£1
    ax = c.axes_mm(x0 + 7.0, yp, 12.0, hp, bg=False)
    est = [1.0, f if k == "sen" else 1.0]
    ax.plot([0, 1], est, color=NC[k], lw=st.LW_DATA, marker="o", ms=2.4, mec="none", zorder=3)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 1.2)
    ax.set_xticks([0, 1], [st.ABS_TICKS[0.0], st.ABS_TICKS[1.0]])
    ax.set_yticks([])
    ax.set_xlabel(st.ABS_XLABEL, labelpad=1)
    ax.set_ylabel("Error SD", labelpad=1)
    ax.set_title(st.TASK_REPRO, fontsize=st.SIZE_TEXT, pad=1.5)
    # Integration: choice variability over sequence length
    ax = c.axes_mm(x0 + 28.0, yp, 24.0, hp, bg=False)
    ax.plot(nn, cvar[k](nn), color=NC[k], lw=st.LW_DATA, zorder=3)
    ax.plot(nn, f ** 2 * cvar[k](nn), color=NC[k], lw=st.LW_DATA, ls=DASH, zorder=3)
    ax.set_xlim(3.5, 12.5)
    ax.set_ylim(0, 7.5)
    ax.set_xticks([4, 8, 12])
    ax.set_yticks([])
    ax.spines["bottom"].set_bounds(4, 12)
    ax.set_xlabel(st.SEQ_LENGTH, labelpad=1)
    ax.set_ylabel("Choice\nvariability", labelpad=1, linespacing=1.0)
    ax.set_title(st.TASK_CATEG, fontsize=st.SIZE_TEXT, pad=1.5)

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

segs = cohort_subexperiments("merged")
bmc = pd.read_csv(ROOT / "results/exploratory/merged/modeling/stats/model_fit_stats/params"
                  / "groupbmc_noise_per_incentive.csv")
sd = merge_sd_inf_long(ROOT, segs)
df_bn = merge_behavior_seqlen(ROOT, segs)
df_pr = pd.read_csv(behav_extra_dir(ROOT, "merged") / "params" / "model_pred_per_seqlen_simple_inf.csv")
for d_ in (df_bn, df_pr):
    d_["participant"] = d_["participant"].astype(str)
    d_["incentive"] = d_["incentive"].astype(float)
    d_["seqlen"] = d_["seqlen"].astype(int)
m_nat = CV.merge_and_preprocess_choice_var_beh_pred(df_bn, df_pr, report_label="pooled natural")
N_D = m_nat["participant"].nunique()
summ_lv = FD.summary_choice_var_pooled_from_merged_cleaned(m_nat)
summ_ef = FD.summary_choice_var_incentive_effect(FD.choice_var_incentive_effect_long_by_seqlen(m_nat))

yr, hr = 76.0, 25.0  # results row: top and axes height (mm)
yt = yr - 7.0  # letters and titles

# --------------------------------------------------------------------------- #
# b  Bayesian model selection
# --------------------------------------------------------------------------- #

c.letter(1, 0.5, yt)
c.title(4.5, yt + 0.2, "Model comparison")
ax = c.axes_mm(10.5, yr + 2.0, 29.0, hr - 4.0)
incs = (LOSS, NEUTRAL, GAIN)
order = ("inf", "sel", "ang")
key = {"inf": "inf", "sel": "sel", "ang": "sen"}
ypos = {LOSS: 2, NEUTRAL: 1, GAIN: 0}
for inc in incs:
    left = 0.0
    for mname in order:
        r = bmc[(bmc["incentive"] == inc) & (bmc["model"] == mname)].iloc[0]
        ax.barh(ypos[inc], r["frequency_mean"], left=left, height=0.62, color=NC[key[mname]], ec="white",
                lw=0.3, zorder=2)
        left += r["frequency_mean"]
ax.set_yticks([ypos[i] for i in incs], [st.INC_TICKS[i] for i in incs])
ax.set_ylim(-0.6, 2.6)
ax.set_xlim(0, 1)
ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
pxp = {inc: float(bmc[(bmc["incentive"] == inc) & (bmc["model"] == "inf")]["pxp"].iloc[0]) for inc in incs}
SD.add("b", bmc[bmc["model"].isin(order)], "group Bayesian model selection among the single-noise models per condition: "
       "estimated model frequencies (frequency_mean, drawn) and protected exceedance probabilities")
assert min(pxp.values()) > 0.99  # the label below states it for every condition
ax.set_xlabel("Model frequency (" + st.sym("sigma", "inf") + ": PXP > .99)", labelpad=1)
ax.set_ylabel(st.INC_XLABEL, labelpad=1)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.legend(handles=[Patch(fc=NC[key[mn]], ec="none", label=st.sym("sigma", key[mn])) for mn in order],
          loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=3, handlelength=0.9, handleheight=0.9,
          columnspacing=0.8, handletextpad=0.3, borderaxespad=0.1)

# --------------------------------------------------------------------------- #
# c  Inference noise
# --------------------------------------------------------------------------- #

xc = 43.0
c.letter(2, xc, yt)
c.title(xc + 4, yt + 0.2, "Inference noise")
d = sd[["participant", "incentive", "sd_inf"]].copy()
d["incentive"] = d["incentive"].astype(float)
d["log_sd_inf"] = np.log(d["sd_inf"])
w = d.pivot(index="participant", columns="incentive", values="sd_inf").dropna()
SD.add_levels("c", w, "sd_inf", "inference noise of the inference-only model per participant and condition, and the "
              "change of its log from neutral", change="logdiff")
ax = c.axes_mm(xc + 8.5, yr, 25.0, hr)
dist.levels_panel(ax, w, log=True, ylabel=st.sym("sigma", "inf"))
log_axis(ax, [0.5, 1, 2, 5])
ax = c.axes_mm(xc + 44.0, yr, 15.0, hr)
dist.effects_panel(ax, dist.incentive_effects(np.log(w)), ylabel=r"Δ log " + st.sym("sigma", "inf"),
                   pvals=FD.paper_holm_p(ROOT, "log_sd_inf"))
dist.pct_axis(ax, "change")

# --------------------------------------------------------------------------- #
# d  Choice variability: data vs model
# --------------------------------------------------------------------------- #

xd = 113.0
c.letter(3, xd, yt)
c.title(xd + 4, yt + 0.2, "Model vs data")
OFF = {LOSS: -0.45, NEUTRAL: 0.0, GAIN: 0.45}
BAND_ALPHA = 0.18  # model ± s.e.m., as in the original panel


def cv_panel(ax, s, beh, beh_se, pred, pred_se, incs):
    """Data: mean ± s.e.m. (dots, offset by condition); model: mean prediction ± s.e.m. (line and band)."""
    for inc in incs:
        q = s[s["incentive"] == inc].sort_values("seqlen")
        col = st.INC_COLORS[inc]
        ax.fill_between(q["seqlen"], q[pred] - q[pred_se], q[pred] + q[pred_se], color=col, alpha=BAND_ALPHA, lw=0,
                        zorder=1.5)
        ax.plot(q["seqlen"], q[pred], color=col, lw=st.LW_DATA, zorder=2 if inc == NEUTRAL else 3)
        ax.errorbar(q["seqlen"] + OFF[inc], q[beh], yerr=q[beh_se], fmt="o", ms=2.4, color=st.shade(col, 0.9),
                    mec="none", elinewidth=st.LW_AXIS, zorder=4)
    ax.set_xlim(2.8, 13.2)
    ax.set_xticks([4, 8, 12])
    ax.spines["bottom"].set_bounds(4, 12)
    ax.set_xlabel(st.SEQ_LENGTH)


ax = c.axes_mm(xd + 6.0, yr, 30.0, hr)
cv_panel(ax, summ_lv, "beh_natural_mean", "beh_natural_sem", "pred_natural_mean", "pred_natural_sem",
         (NEUTRAL, LOSS, GAIN))
ax.set_ylim(0, None)
ax.set_ylabel("Choice variability")
ax.legend(handles=[Line2D([], [], color=st.INC_COLORS[i], lw=st.LW_DATA, marker="o", ms=2.4, mec="none",
                          label=st.INC_TICKS[i]) for i in (LOSS, NEUTRAL, GAIN)],
          title=st.INC_XLABEL, loc="upper left", bbox_to_anchor=(-0.02, 1.04), handlelength=1.4,
          labelspacing=0.15)
ax = c.axes_mm(xd + 45.0, yr, 23.5, hr)
cv_panel(ax, summ_ef, "beh_red_mean", "beh_red_sem", "pred_red_mean", "pred_red_sem", (LOSS, GAIN))
dist.zero_line(ax)
ax.set_yticks([0, 0.5, 1.0], ["0", "0.5", "1"])
ax.set_ylabel("Reduction (£0 − incentive)")
ax.legend(handles=[Line2D([], [], color="k", marker="o", ms=2.4, mec="none", lw=0, label="data"),
                   (Patch(fc="k", alpha=BAND_ALPHA, ec="none"), Line2D([], [], color="k", lw=st.LW_DATA))],
          labels=["data", "model"],
          loc="upper left", bbox_to_anchor=(-0.02, 1.04), handlelength=1.4, labelspacing=0.15)

if os.environ.get("FIG_DEBUG"):  # horizontal extent of each axes and of its labels (mm), for layout
    c.fig.canvas.draw()
    rr = c.fig.canvas.get_renderer()
    k_ = 25.4 / c.fig.dpi
    for a in c.fig.axes[1:]:
        p_, t_ = a.get_window_extent(rr), a.get_tightbbox(rr)
        print(f"axes {p_.x0 * k_:6.1f}-{p_.x1 * k_:6.1f} | labels {t_.x0 * k_:6.1f}-{t_.x1 * k_:6.1f} "
              f"| y {H - p_.y1 * k_:5.1f}")
SD.add("d", summ_lv, f"choice variability per condition and sequence length: data and inference-only model, mean and "
       f"s.e.m. over participants (N = {N_D})", part="levels")
SD.add("d", summ_ef, f"reduction of choice variability (neutral − incentive): data and model, mean and s.e.m. "
       f"(N = {N_D})", part="reduction")
SD.add("d", m_nat, "per participant, condition and sequence length: the values summarised above", part="participants")
st.save(fig := c.fig, OUT / "figures" / "fig2")
SD.write()
print("saved", OUT / "figures" / "fig2.svg", "| N in d:", N_D)
