"""Supplementary Figure 7: common biases.

Panels:
    a  Stimulus Reproduction: clockwise bias against the orientation (cardinal bias) and against the difference
       from the previous orientation (serial bias), per condition; mean absolute bias per condition and its
       change from £0
    b  Sequence Categorization: evidence weight per sample position, counted from the end of the sequence
    c  Sequence Categorization: choice-history probit coefficients per condition and their change from £0
Stars: Holm-corrected tests of Supplementary Table 6 (none reaches p < .05).

Inputs (results/exploratory/merged/modeling/stats/bias_stats/params/): estimation_ksr_curves_long.csv,
estimation_mean_abs_dev.csv, temporal_beta_long.csv and probit_carryover_coefs_long.csv; stars from
results/exploratory/merged/paper_tables/paper_tables.json (supp → biases).
Output: plots/exploratory/figures/sfig7.{svg,pdf,png}, at print size (180 mm wide).

Usage (from the repository root):
    python scripts/exploratory/figures/sfig7.py
"""

from __future__ import annotations

import figpaths  # noqa: F401  (paths first)

import pandas as pd
from matplotlib.lines import Line2D

from motinf.exploratory import figure_data as P  # reading and Holm p
from motinf.exploratory.figlib import dist, figstyle as st
from motinf.exploratory.figlib.canvas import Canvas
from motinf.exploratory.figlib.source_data import SourceData

st.apply()
ROOT, OUT = figpaths.DATA_ROOT, figpaths.OUT_ROOT
SD = SourceData("sfig7", OUT)
PAR = ROOT / "results/exploratory/merged/modeling/stats/bias_stats/params"
LOSS, NEUTRAL, GAIN = st.LOSS, st.NEUTRAL, st.GAIN
INCS = (LOSS, NEUTRAL, GAIN)

curves = pd.read_csv(PAR / "estimation_ksr_curves_long.csv")
mad = pd.read_csv(PAR / "estimation_mean_abs_dev.csv")
beta = pd.read_csv(PAR / "temporal_beta_long.csv")
probit = pd.read_csv(PAR / "probit_carryover_coefs_long.csv")
probit = probit[probit["fit_ok"].astype(bool)]
for d in (curves, mad, beta, probit):
    d["incentive"] = d["incentive"].astype(float)


def mean_sem(d, x, y):
    return d.groupby(["incentive", x])[y].agg(["mean", "sem"]).reset_index()


def mean_sem_lines(ax, d, x, y):
    """Mean ± s.e.m. across participants per condition (seaborn's errorbar="se" in the original)."""
    g = mean_sem(d, x, y)
    for inc in INCS:
        s = g[g["incentive"] == inc]
        col = st.INC_COLORS[inc]
        ax.fill_between(s[x], s["mean"] - s["sem"], s["mean"] + s["sem"], color=col, alpha=0.2, lw=0,
                        zorder=2 + inc)
        ax.plot(s[x], s["mean"], color=col, lw=st.LW_DATA, zorder=3 + inc)


def wide(d, val):
    return d.pivot(index="participant", columns="incentive", values=val).dropna()


W, H = st.LAYOUT_DOUBLE, 124.5  # layout mm; saved at st.DOUBLE wide (st.page_scale)
c = Canvas(W, H)
yt = 1.6
REF = dict(color=st.REF_LINE, lw=st.LW_THIN, ls=(0, (3, 2)), zorder=1)

# a  orientation biases
c.letter(0, 0.5, yt)
c.title(4.5, yt + 0.2, f"Orientation biases in {st.TASK_REPRO}")
SUB = [("cardinal", "Cardinal bias", "Stimulus orientation (°)", (0, 180), [0, 90, 180]),
       ("serial", "Serial bias", "Current − previous orientation (°)", (-90, 90), [-90, 0, 90])]
ya0, AH, DY = 10.5, 23.0, 35.0
for r, (task, name, xlab, xlim, xt) in enumerate(SUB):
    y = ya0 + r * DY
    c.text(4.5, y - 3.4, name, fontweight="bold", va="center")
    ax = c.axes_mm(14.0, y, 44.0, AH)
    mean_sem_lines(ax, curves[curves["task"] == task], "angle", "bias")
    ax.axhline(0, **REF)
    ax.set_xlim(*xlim)
    ax.set_xticks(xt)
    ax.set_ylim(-7, 7)
    ax.set_yticks([-6, 0, 6], ["−6", "0", "6"])
    ax.set_xlabel(xlab)
    ax.set_ylabel("Clockwise bias (°)")
    w = wide(mad[mad["task"] == task], "mean_abs_dev")
    SD.add("a", mean_sem(curves[curves["task"] == task], "angle", "bias"), f"{name.lower()}: clockwise bias (deg) "
           f"against the {xlab[:-4].lower()} (angle, deg) per condition, mean and s.e.m. over participants",
           part=f"{task}_curves")
    SD.add_levels("a", w, "mean_abs_dev", f"{name.lower()}: mean absolute bias (deg) per participant and condition, "
                  "and its change from neutral", part=f"{task}_mean_abs_bias")
    ax = c.axes_mm(70.0, y, 25.0, AH)
    dist.levels_panel(ax, w, ylabel="Mean |bias| (°)")
    ax = c.axes_mm(104.0, y, 16.0, AH)
    dist.effects_panel(ax, dist.incentive_effects(w), ylabel="Δ mean |bias| (°)",
                       pvals=P.paper_bias_holm_p(ROOT, task))

# b  recency
xb = 129.0
c.letter(1, xb, yt)
c.title(xb + 4.0, yt + 0.2, f"Recency in {st.TASK_CATEG}")
ax = c.axes_mm(xb + 11.0, ya0 - 2.0, 41.0, DY + AH + 2.0)
n_lag = int(beta["position"].max()) + 1
beta["pos"] = beta["position"].astype(int) - n_lag
mean_sem_lines(ax, beta, "pos", "beta")
SD.add("b", mean_sem(beta, "pos", "beta"), "evidence weight per sample position (counted from the end of the "
       "sequence) and condition, mean and s.e.m. over participants")
ax.axhline(0, **REF)
ax.set_xlim(-12.4, -0.6)
ax.set_xticks([-12, -8, -4, -1], ["−12", "−8", "−4", "−1"])
ax.set_ylim(-0.03, 0.85)
ax.set_xlabel("Sample position (from the end)")
ax.set_ylabel("Evidence weight (β)")
ax.legend(handles=[Line2D([], [], color=st.INC_COLORS[i], lw=st.LW_DATA * 1.5, label=st.INC_TICKS[i]) for i in INCS],
          title=st.INC_XLABEL, loc="upper left", alignment="left", handlelength=1.4)

# c  choice history
yc = ya0 + DY + AH + 11.5
c.letter(2, 0.5, yc)
c.title(4.5, yc + 0.2, f"Choice history in {st.TASK_CATEG} (probit regression)")
COEF = [("b_0", "Category preference"), ("b_prev_stim", "Previous category"), ("b_prev_resp", "Previous choice"),
        ("b_prev_key", "Previous key")]
yp = yc + 9.0
groups, pv = [], {}
for k, (coef, name) in enumerate(COEF):
    w = wide(probit[probit["coef"] == coef], "value")
    SD.add_levels("c", w, coef, f"choice-history probit coefficient ({name.lower()}) per participant and condition, "
                  "and its change from neutral (drawn on the right as mean ± 95% CI)", part=coef)
    ax = c.axes_mm(14.0 + k * 30.5, yp, 23.0, 26.0)
    dist.levels_panel(ax, w, ylabel="Probit coefficient" if k == 0 else None, xlabel=st.INC_XLABEL if k == 0 else "")
    ax.axhline(0, **REF)
    ax.locator_params(axis="y", nbins=4)
    c.text(14.0 + k * 30.5 + 11.5, yp - 1.2, name, ha="center", va="bottom", fontsize=st.SIZE_TEXT)
    groups.append((coef, dist.incentive_effects(w)))
    pv.update({(coef, i): p for i, p in P.paper_bias_holm_p(ROOT, coef).items()})
ax = c.axes_mm(146.0, yp, 35.0, 26.0)
for k, (coef, eff) in enumerate(groups):
    for inc, off in ((LOSS, -0.16), (GAIN, 0.16)):
        m, lo, hi = dist.t_ci(eff[inc])
        ax.plot([k + off] * 2, [lo, hi], color=st.INC_COLORS[inc], lw=1.0, solid_capstyle="butt", zorder=3)
        ax.plot([k + off], [m], "o", ms=3.0, color=st.INC_COLORS[inc], mec="white", mew=0.4, zorder=4)
        s = dist.p_stars(pv[(coef, inc)])
        if s:
            ax.text(k + off, hi, s, ha="center", va="bottom", fontsize=st.SIZE_TEXT)
ax.axhline(0, **REF)
ax.set_xlim(-0.5, len(groups) - 0.5)
ax.margins(y=0.08)
ax.set_xticks(range(len(groups)), ["Category\npref.", "Prev.\ncategory", "Prev.\nchoice", "Prev.\nkey"])
ax.tick_params(axis="x", length=0)
ax.locator_params(axis="y", nbins=4)
ax.set_ylabel("Δ probit coefficient (vs £0)")

st.save(c.fig, OUT / "figures" / "sfig7")
SD.write()
nn = {t: len(wide(mad[mad["task"] == t], "mean_abs_dev")) for t in ("cardinal", "serial")}
print("saved sfig7", f"{W:.0f} x {H:.0f} mm", "N mad:", nn, "N beta:", beta["participant"].nunique(),
      "N probit:", {k: len(wide(probit[probit['coef'] == k], 'value')) for k, _ in COEF})
print("Holm p:", {k: round(v, 3) for k, v in pv.items()})
