"""Figure 5: response times.

Panels:
    a  median response time per condition and its change from £0 (ms), both tasks
    b  Sequence Categorization: response time against the evidence toward the chosen option, per sequence length
       and condition (binned data and linear fits), with the fits' intercepts and slopes and their differences
       from £0 (Supplementary Table 8)
    c  speed-accuracy trade-off: noise in fast and slow trials (median split), fast − slow per condition and its
       change with incentives, both tasks (Supplementary Table 9)

Inputs: results/exploratory/merged/behavior/stats/rt_analysis/*.csv; results/exploratory/merged/paper_tables/
paper_tables.json (stars of a: Supplementary Table 7).
Output: plots/exploratory/figures/fig5.{svg,pdf,png}, at print size (180 mm wide).

Usage (from the repository root):
    python scripts/exploratory/figures/fig5.py
"""

from __future__ import annotations

import os

import figpaths  # noqa: F401  (paths first)

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

from motinf.exploratory import figure_data as P  # Fig. 5 computations
from motinf.exploratory.figlib import dist, figstyle as st
from motinf.exploratory.figlib.canvas import Canvas
from motinf.exploratory.figlib.source_data import SourceData
from motinf.exploratory.rt_analysis import rt_stats_dir

st.apply()
ROOT, OUT = figpaths.DATA_ROOT, figpaths.OUT_ROOT
W, H = st.LAYOUT_DOUBLE, 128.5  # layout mm; saved at st.DOUBLE wide (st.page_scale)
LOSS, NEUTRAL, GAIN = st.LOSS, st.NEUTRAL, st.GAIN
INC = (LOSS, NEUTRAL, GAIN)
BY_CONTRAST = {"loss_minus_neutral": LOSS, "gain_minus_neutral": GAIN}
SD = rt_stats_dir(ROOT, "merged")
SRC = SourceData("fig5", OUT)


def rd(name):
    return pd.read_csv(SD / name)


c = Canvas(W, H)
HAX = 24.0  # height of the distribution plots (mm)
TASK_X = {"estim": -2.5, "inf": 92.0}  # left edge of each task block (labels start 3.5 mm to its right)
TASK_NAME = {"estim": st.TASK_REPRO, "inf": st.TASK_CATEG}


def task_pair(task, y, lv, eff, *, lv_ylabel, eff_ylabel, lv_pvals=None, eff_pvals=None, lv_kind="levels",
              scale=1.0):
    """Per-condition plot | change-from-£0 plot for one task, at the task's block position."""
    x0 = TASK_X[task]
    ax = c.axes_mm(x0 + 13.0, y, 36.0, HAX)
    if lv_kind == "levels":
        dist.levels_panel(ax, lv, ylabel=lv_ylabel)
        pv = lv_pvals or {}
        for k, x in zip(INC, range(3)):
            dist.stars_at(ax, x, pv.get(k, np.nan))
    else:  # per-condition differences around zero
        dist.effects_panel(ax, lv, contrasts=INC, names=st.INC_TICKS, pvals=lv_pvals, ylabel=lv_ylabel,
                           xlabel=st.INC_XLABEL)
    ax2 = c.axes_mm(x0 + 62.0, y, 27.0, HAX)
    dist.effects_panel(ax2, eff, ylabel=eff_ylabel, pvals=eff_pvals)
    return ax, ax2


# --------------------------------------------------------------------------- #
# a  Median RT
# --------------------------------------------------------------------------- #

c.letter(0, 0.5, 1.6)
c.title(4.5, 1.8, "Incentives slow responses in both tasks")
ya = 1.6 + 9.2  # rows a and c: task names 3.8 mm and plots 9.2 mm below the letter
df_mean = rd("rt_median_by_incentive.csv")
for task, dv, eff_file, measure in (("estim", "rt_estim_median", "rt_effect_estim.csv", "estim_rt"),
                                    ("inf", "rt_median", "rt_effect_inf.csv", "categ_rt")):
    c.text(TASK_X[task] + 7.0, 1.6 + 3.8, TASK_NAME[task], fontweight="bold", fontsize=st.SIZE_LABEL)
    d = df_mean[["participant", "incentive", dv]].dropna().copy()
    d["incentive"] = d["incentive"].astype(float)
    wide = d.pivot(index="participant", columns="incentive", values=dv).dropna()
    eff = {BY_CONTRAST[k]: 1000.0 * g["effect"].to_numpy(float) for k, g in rd(eff_file).groupby("contrast")}
    SRC.add_levels("a", wide, dv, f"{TASK_NAME[task]}: median response time (s) per participant and condition, and "
                   "its change from neutral (s; drawn in ms)", part=task)
    task_pair(task, ya, wide, eff, lv_ylabel="Median RT (s)", eff_ylabel="Δ median RT (ms)",
              eff_pvals=P.paper_holm_p(ROOT, measure))

# --------------------------------------------------------------------------- #
# b  Integration: RT and evidence
# --------------------------------------------------------------------------- #

yb = 43.3
c.letter(1, 0.5, yb)
c.title(4.5, yb + 0.2, f"{st.TASK_CATEG}: slower when the evidence is weak, more so under incentives")
coefs = rd("rt_evidence_cell_linear_coefs.csv")
bins = rd("rt_evidence_cell_linear_bins.csv")
tests = rd("rt_evidence_cell_linear_tests.csv")
curves = P.cell_linear_curves(coefs)
pts = P.cell_linear_points(bins)
seqlens = sorted(coefs["seqlen"].unique())
# as in the original panel: hue = incentive (the paper's red / grey / green), lightness = sequence length
SEQ_TONE = {4.0: lambda col: st.shade(col, 0.8), 8.0: lambda col: col, 12.0: lambda col: st.tint(col, 0.5)}


def tone(inc, s):
    return SEQ_TONE[float(s)](st.INC_COLORS[float(inc)])


def is_dark(rgb):
    r, g, b = to_rgb(rgb)
    return 0.299 * r + 0.587 * g + 0.114 * b < 0.5


# all nine fits in one plot
ax = c.axes_mm(10.5, yb + 4.9, 60.0, 31.0)
for s in seqlens:
    for inc in (NEUTRAL, GAIN, LOSS):
        col = tone(inc, s)
        gx, gy, glo, ghi = curves[(s, inc)]
        ax.fill_between(gx, glo, ghi, color=col, alpha=0.15, lw=0, zorder=1)
        ax.plot(gx, gy, color=col, lw=st.LW_DATA, zorder=2 if inc == NEUTRAL else 3)
        p = pts[(pts["seqlen"] == s) & (pts["incentive"] == inc)]
        yv = np.exp(p["m"])
        ax.errorbar(p["x"], yv, yerr=[yv - np.exp(p["m"] - p["se"]), np.exp(p["m"] + p["se"]) - yv], fmt="o", ms=2.0,
                    color=col, mec="none", elinewidth=st.LW_AXIS, zorder=4)
ax.axvline(0, color=st.REF_LINE, lw=st.LW_THIN, ls=(0, (1, 1.5)), zorder=0)
ax.set_xlim(-4.6, 6.6)
ax.set_xticks([-4, -2, 0, 2, 4, 6], ["−4", "−2", "0", "2", "4", "6"])
ax.set_xlabel("Evidence toward the choice (Σ log LR)")
ax.set_ylabel("RT (s)")
leg_inc = ax.legend(handles=[Line2D([], [], color=tone(i, 4.0), lw=st.LW_DATA, label=st.INC_TICKS[i])
                             for i in (LOSS, NEUTRAL, GAIN)], title=st.INC_XLABEL, loc="upper right",
                    bbox_to_anchor=(1.0, 1.04), handlelength=1.3, labelspacing=0.15)
ax.add_artist(leg_inc)
ax.legend(handles=[Line2D([], [], color=tone(NEUTRAL, s), lw=st.LW_DATA, label=f"{s:g}") for s in seqlens],
          title="Gratings", loc="upper right", bbox_to_anchor=(0.66, 1.04), handlelength=1.3, labelspacing=0.15)

# intercepts and slopes of the fits, and their incentive − neutral differences
Wd = P.cell_linear_wide(coefs)
n_part = len(Wd["intercept"])
COLS = [("lvl", "intercept", "Intercept\n(RT at 0, s)", [0, 0.5, 1.0], ["0", "0.5", "1"]),
        ("lvl", "slope", "Slope\n(log RT per log LR)", [-0.1, 0], ["−0.1", "0"]),
        ("diff", "intercept", "Δ intercept\n(ms)", [0, 20, 40], ["0", "20", "40"]),
        ("diff", "slope", "Δ slope\n(log RT per log LR)", [-0.01, 0], ["−0.01", "0"])]
ROWS = (NEUTRAL, GAIN, LOSS)
ROW_NAME = {NEUTRAL: "neutral", GAIN: "gain", LOSS: "loss"}
XC = [85.5, 108.5, 136.5, 159.5]  # left edge of each column (mm)
WC, HR, GAP = 20.0, 8.6, 1.6  # column width, row height, gap between rows
yr0 = yb + 12.3
vals = {}
for r_, inc in enumerate(ROWS):
    for k, (kind, coef, _, _, _) in enumerate(COLS):
        if kind == "diff" and inc == NEUTRAL:
            continue
        for i, s in enumerate(seqlens):
            x = Wd[coef][(inc, s)].to_numpy(float)
            if kind == "lvl":
                m, se = x.mean(), x.std(ddof=1) / np.sqrt(n_part)
                f = np.exp if coef == "intercept" else (lambda v: v)
                v, lo_, hi_ = f(m), f(m - se), f(m + se)
                pv = P.cell_p(tests, "slope_vs_zero", "slope", ROW_NAME[inc], s) if coef == "slope" else np.nan
            else:
                ref = Wd[coef][(NEUTRAL, s)].to_numpy(float)
                v, hw = P.cell_diff_units(coef, ref.mean(), x - ref)
                lo_, hi_ = v - hw, v + hw
                pv = P.cell_p(tests, "incentive_vs_neutral", coef, f"{ROW_NAME[inc]}_minus_neutral", s)
            vals[(r_, k, i)] = (v, lo_, hi_, pv)
for k, (kind, coef, label, ticks, tlabs) in enumerate(COLS):
    lo_all = min(min(vals[(r_, k, i)][1] for r_ in range(3) for i in range(3) if (r_, k, i) in vals), min(ticks))
    hi_all = max(max(vals[(r_, k, i)][2] for r_ in range(3) for i in range(3) if (r_, k, i) in vals), max(ticks))
    pad = 0.02 * (hi_all - lo_all)
    for r_, inc in enumerate(ROWS):
        if kind == "diff" and inc == NEUTRAL:
            continue
        ax = c.axes_mm(XC[k], yr0 + r_ * (HR + GAP), WC, HR)
        for i, s in enumerate(seqlens):
            v, lo_, hi_, pv = vals[(r_, k, i)]
            col = tone(inc, s)
            ax.barh(i, v, 0.82, color=col, zorder=2)
            ax.errorbar(v, i, xerr=[[v - lo_], [hi_ - v]], fmt="none", color="k", lw=st.LW_AXIS, capsize=0, zorder=3)
            sg = dist.p_stars(pv)
            if sg:  # inside the bar, next to the zero line
                right = v >= 0
                ax.annotate(sg, (0, i), xytext=(1.5 if right else -1.5, -1.6), textcoords="offset points",
                            ha="left" if right else "right", va="center", fontsize=st.SIZE_MIN,
                            color="white" if is_dark(col) else "k", zorder=5)
        ax.axvline(0, color="k", lw=st.LW_AXIS, zorder=4)
        ax.set_xlim(lo_all - pad, hi_all + pad)
        ax.set_ylim(len(seqlens) - 0.5, -0.5)
        for sp in ("top", "right", "left", "bottom"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        if k == 0:
            ax.set_yticks(range(len(seqlens)), [f"{s:g}" for s in seqlens])
            ax.set_ylabel(ROW_NAME[inc].capitalize(), labelpad=2.0)
        else:
            ax.set_yticks([])
        top = (r_ == 0) if kind == "lvl" else (r_ == 1)
        if top:  # the value axis, once, above the column
            ax.spines["top"].set_visible(True)
            ax.spines["top"].set_bounds(min(ticks), max(ticks))
            ax.xaxis.set_label_position("top")
            ax.tick_params(axis="x", top=True, labeltop=True)
            ax.set_xticks(ticks, tlabs)
            ax.set_xlabel(label, linespacing=1.0, labelpad=2.0)
c.text(XC[0] - 9.0, yr0 + 1.5 * (HR + GAP) - GAP / 2, "Gratings", rotation=90, ha="center", va="center",
       fontsize=st.SIZE_TEXT)
SRC.add("b", pts, "binned data: mean log response time (m, s.e.m. se) per bin of evidence toward the choice (x), "
        "per sequence length and condition (drawn as exp(m))", part="bins")
SRC.add("b", pd.concat([pd.DataFrame({"seqlen": s_, "incentive": i_, "evidence": gx, "rt_fit": gy, "rt_fit_lo": glo,
                                      "rt_fit_hi": ghi}) for (s_, i_), (gx, gy, glo, ghi) in curves.items()],
                       ignore_index=True), "linear fits of log response time on evidence (mean and s.e.m. band, "
        "in s)", part="fits")
SRC.add("b", coefs, "per participant, sequence length and condition: intercept and slope of the fits",
        part="coefficients")
SRC.add("b", pd.DataFrame([{"condition": ROW_NAME[ROWS[r_]], "quantity": ("" if COLS[k][0] == "lvl" else "delta_")
                            + COLS[k][1], "seqlen": seqlens[i], "value": v, "lo": lo_, "hi": hi_, "p": pv}
                           for (r_, k, i), (v, lo_, hi_, pv) in sorted(vals.items())]),
        "bars on the right: intercept (s) and slope (mean ± s.e.m.) and their differences from neutral (intercept "
        "in ms), with the p values of the stars", part="bars")

# --------------------------------------------------------------------------- #
# c  Noise in fast vs slow trials
# --------------------------------------------------------------------------- #

yc = 88.0
c.letter(2, 0.5, yc)
c.title(4.5, yc + 0.2, "Speed-accuracy trade-off")
spec = {"estim": ("log_sd_estim", "(log error SD)"), "inf": ("sd_inf_log", r"(log $\sigma_{\mathrm{inf}}$)")}
for task, (metric, unit) in spec.items():
    c.text(TASK_X[task] + 7.0, yc + 3.8, TASK_NAME[task], fontweight="bold", fontsize=st.SIZE_LABEL)
    dfs = rd(f"diff_fast_minus_slow_{metric}.csv")
    inc_ = dfs["incentive"].astype(float)
    lv = {k: dfs.loc[inc_ == k, "diff"].to_numpy(float) for k in INC}
    t_lv = rd(f"ttest_noise_fast_minus_slow_by_incentive_{metric}.csv")
    pv_lv = {k: P.holm_p(t_lv, "incentive", k) for k in INC}
    de = rd(f"diff_fast_minus_slow_incentive_effect_{metric}.csv")
    eff = {BY_CONTRAST[k]: g["diff"].to_numpy(float) for k, g in de.groupby("contrast")}
    SRC.add("c", dfs, f"{TASK_NAME[task]}: noise ({'log error SD' if task == 'estim' else 'log inference noise'}), "
            "fast − slow trials, per participant and condition",
            part=f"{task}_fast_minus_slow")
    SRC.add("c", de, f"{TASK_NAME[task]}: change of the fast − slow difference with incentives, per participant",
            part=f"{task}_incentive_effect")
    t_e = rd(f"ttest_split_{metric}.csv")
    pv_e = {BY_CONTRAST[k]: P.holm_p(t_e, "contrast", k) for k in BY_CONTRAST}
    task_pair(task, yc + 9.2, lv, eff, lv_kind="effects", lv_ylabel="Noise, fast − slow\n" + unit,
              eff_ylabel="Incentive effect,\nfast − slow", lv_pvals=pv_lv, eff_pvals=pv_e)

if os.environ.get("FIG_DEBUG"):
    c.fig.canvas.draw()
    rr = c.fig.canvas.get_renderer()
    k_ = 25.4 / c.fig.dpi
    for a in c.fig.axes[1:]:
        p_, t_ = a.get_window_extent(rr), a.get_tightbbox(rr)
        print(f"axes {p_.x0 * k_:6.1f}-{p_.x1 * k_:6.1f} | labels {t_.x0 * k_:6.1f}-{t_.x1 * k_:6.1f} "
              f"| y {H - p_.y1 * k_:5.1f}-{H - p_.y0 * k_:5.1f} | labels y {H - t_.y1 * k_:5.1f}-{H - t_.y0 * k_:5.1f}")
st.save(c.fig, OUT / "figures" / "fig5")
SRC.write()
print("saved", OUT / "figures" / "fig5.svg", "| N:", df_mean["participant"].nunique())
