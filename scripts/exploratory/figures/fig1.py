"""Figure 1: task design and behavioral effects of incentives.

Panels:
    a  design: the two sessions and the incentive cues (drawn)
    b  Stimulus Reproduction trial, and payoff as a function of the error in Experiments 1 and 2 (drawn)
    c  orientation distributions of the two categories and of the Stimulus Reproduction task; evidence (log LR)
       carried by one grating (drawn)
    d  Sequence Categorization trial (drawn)
    e  Stimulus Reproduction: error SD per condition and its change from £0
    f  Sequence Categorization: proportion of choices consistent with the evidence against the absolute summed
       evidence (binned data, mean of the participants' probit fits)
    g  Sequence Categorization: evidence sensitivity per condition and its change from £0

Inputs: results/prereg/<exp>/behavior/data_mean_inc.csv (e, g);
results/exploratory/merged/behavior/stats/behav_extra/params/psychometric_consistency_{bins,curves}.csv (f);
results/exploratory/merged/paper_tables/paper_tables.json (stars of e and g: Table 2).
Output: plots/exploratory/figures/fig1.{svg,pdf,png}, at print size (180 mm wide).

Usage (from the repository root):
    python scripts/exploratory/figures/fig1.py
"""

from __future__ import annotations

import figpaths  # noqa: F401  (paths first)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from motinf.exploratory.figlib import dist, figstyle as st
from motinf.exploratory.figlib.canvas import Canvas, gabor
from motinf.exploratory.figlib.source_data import SourceData
from motinf.exploratory.figure_data import paper_holm_p

st.apply()
ROOT, OUT = figpaths.DATA_ROOT, figpaths.OUT_ROOT
SD = SourceData("fig1", OUT)
W, H = st.LAYOUT_DOUBLE, 108.0  # layout mm; saved at st.DOUBLE wide (st.page_scale)
LOSS, NEUTRAL, GAIN = st.LOSS, st.NEUTRAL, st.GAIN
PINK, BLUE = "#e377c2", "#4a90d9"  # category colours of the Integration task (pink −45°, blue +45°)
SCREEN = "#7f7f7f"
EX_INC = GAIN  # incentive of the example trials
# task screens (b and d share the geometry): 4:3 rectangles, same overlap and vertical step
SW, SH = 18.0, 13.5
OV, DY = 2.5, 2.2
DX = SW - OV

# --------------------------------------------------------------------------- #
# Canvas (millimetres, origin top-left; figlib.canvas)
# --------------------------------------------------------------------------- #

canvas = Canvas(W, H)
fig, cv = canvas.fig, canvas.cv
axes_mm, text, letter, title, rbox, image = (canvas.axes_mm, canvas.text, canvas.letter, canvas.title, canvas.rbox,
                                               canvas.image)

# --------------------------------------------------------------------------- #
# Stimuli
# --------------------------------------------------------------------------- #


def color_wheel(n=181, r_in=0.89):
    """Thin ring whose colour goes from pink (−45°) to blue (+45°) with the axial angle."""
    y, x = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]
    r = np.hypot(x, y)
    phi = np.arctan2(-y, x)
    w = 0.5 * (1 + np.cos(2 * (phi - np.deg2rad(45))))
    c1, c2 = np.array(plt.matplotlib.colors.to_rgb(BLUE)), np.array(plt.matplotlib.colors.to_rgb(PINK))
    rgb = w[..., None] * c1 + (1 - w[..., None]) * c2
    return np.dstack([rgb, ((r <= 1) & (r >= r_in)).astype(float)])


def screen(x, y, w=SW, h=SH, z=2, inc=EX_INC):
    """Task screen (w × h): grey, framed in the incentive colour, incentive cue at the top. Returns the
    centre of the stimulus area."""
    rbox(x, y, w, h, fc=SCREEN, ec=st.INC_COLORS[inc], lw=0.9, r=0.8, z=z)
    cue(x + w / 2, y + 0.14 * h, inc, r=0.075 * h, z=z + 0.2)
    return x + w / 2, y + h / 2 + 0.06 * h


def fixation(cx, cy, z, r=0.45):
    cv.add_patch(Circle((cx, cy), r, fc="k", ec="none", zorder=z))


def cue(cx, cy, inc, r=1.7, z=6, fs=None):
    cv.add_patch(Circle((cx, cy), r, fc="white", ec=st.INC_COLORS[inc], lw=0.8, zorder=z))
    cv.text(cx, cy, "£0" if inc == NEUTRAL else "£1", fontsize=fs or st.SIZE_MIN, ha="center", va="center",
            zorder=z + 0.1)


def bar(cx, cy, theta_deg, half, color, lw, z):
    """Line segment at theta (clockwise from vertical) on the canvas (y points down)."""
    t = np.deg2rad(theta_deg)
    cv.plot([cx - half * np.sin(t), cx + half * np.sin(t)], [cy + half * np.cos(t), cy - half * np.cos(t)],
            color=color, lw=lw, solid_capstyle="round", zorder=z)


def response_screen_integration(x, y, z, w=SW, h=SH):
    """As in the task: category buttons (bars at −45° pink / +45° blue) over the arrow keys, colour wheel
    with the '?' prompt in the centre."""
    cx, cy = screen(x, y, w, h, z=z)
    image(color_wheel(), cx, cy, 0.55 * h, z=z + 0.1)
    rbox(cx - 0.08 * h, cy - 0.08 * h, 0.16 * h, 0.16 * h, fc="0.35", r=0.3, z=z + 0.2)
    cv.add_patch(Circle((cx, cy), 0.055 * h, fc="k", ec="none", zorder=z + 0.3))
    cv.text(cx, cy, "?", color="white", fontsize=st.SIZE_MIN, ha="center", va="center", fontweight="bold",
            zorder=z + 0.4)
    for side, col, th in ((-1, PINK, -45), (1, BLUE, 45)):
        bx = cx + side * 0.33 * w
        b = 0.19 * h
        rbox(bx - b / 2, cy - 0.24 * h, b, b, fc="#bdbdbd", ec="0.3", lw=0.3, r=0.3, z=z + 0.2)
        bar(bx, cy - 0.24 * h + b / 2, th, 0.06 * h, col, 1.3, z + 0.3)
        rbox(bx - b / 2, cy + 0.04 * h, b, b, fc="0.25", ec="0.1", lw=0.3, r=0.3, z=z + 0.2)
        cv.plot([bx], [cy + 0.04 * h + b / 2], marker="<" if side < 0 else ">", ms=2.0, color="white", lw=0,
                zorder=z + 0.3)


# --------------------------------------------------------------------------- #
# a  Design: sessions as vertical blocks, incentive cues beside them
# --------------------------------------------------------------------------- #

letter(0, 0.5, 1.6)
title(4.5, 1.8, "Design")
ya, ha_ = 8.6, 22.5  # top and height of the blocks
bw, gap = 6.8, 1.6
x = 4.5
blocks = {"E": (st.TASK_REPRO.replace(" ", "\n"), "72 trials", "#f2f2f2", "-"),  # task names on two lines
          "I": (st.TASK_CATEG.replace(" ", "\n"), "432 trials", "#f2f2f2", "-"),
          "Q": ("Questionnaire", "BIS/BAS, SPSRQ-RC", "white", (0, (2, 1)))}
for k, (sess, seq) in enumerate((("Session 1", "EI"), ("Session 2", "EIQ"))):
    text(x, ya - 1.2, sess, fontweight="bold", va="bottom")
    for b in seq:
        name, sub, fc, ls = blocks[b]
        two = "\n" in name
        rbox(x, ya, bw, ha_, fc=fc, ec="0.35", lw=st.LW_AXIS, r=0.6, ls=ls)
        text(x + bw / 2 - (1.15 if two else 0.9), ya + ha_ / 2, name, rotation=90, ha="center", va="center",
             linespacing=1.0)
        text(x + bw / 2 + (2.05 if two else 1.35), ya + ha_ / 2, sub, rotation=90, ha="center", va="center",
             fontsize=st.SIZE_MIN)
        x += bw + gap
    if k == 0:
        text(x + 2.6, ya + ha_ / 2, "…", ha="center", va="center")
        x += 7.0
# incentive cues
xi = x + 3.0
text(xi, ya - 1.2, "Incentive cue", fontweight="bold", va="bottom")
text(xi, ya + 0.4, "mini-blocks of 6 trials", fontsize=st.SIZE_MIN)
for j, (inc, lab, pay) in enumerate(((GAIN, "+1  gain", "win up to £1"), (NEUTRAL, "0  neutral", "no stake"),
                                     (LOSS, "−1  loss", "avoid losing up to £1"))):
    cy = ya + 6.6 + 6.0 * j
    cue(xi + 2.2, cy, inc, r=2.2)
    text(xi + 5.4, cy - 0.6, lab, va="bottom")
    text(xi + 5.4, cy - 0.2, pay, va="top", fontsize=st.SIZE_MIN)
SD.note("a", "design schematic (no data)")

# --------------------------------------------------------------------------- #
# b  Estimation trial + payoff
# --------------------------------------------------------------------------- #

xb0 = 84.0
letter(1, xb0 - 4, 1.6)
title(xb0, 1.8, f"{st.TASK_REPRO} task")
yb0 = ya  # first screen level with the blocks of a
rr = 0.3 * SH  # response circle radius
for i, (lab, kind) in enumerate((("660 ms", "fix"), ("100 ms", "gabor"), ("until response\n(≤ 7 s, Exp. 2)", "resp"))):
    x, y = xb0 + i * DX, yb0 + i * DY
    cx, cy = screen(x, y, z=2 + i)
    cv.add_patch(Circle((cx, cy), rr, fc="none", ec="k", lw=0.6, zorder=3 + i))
    if kind == "fix":
        fixation(cx, cy, 3.2 + i)
    elif kind == "gabor":
        image(gabor(35, seed=1), cx, cy, 1.9 * rr, z=3.1 + i)
    else:
        bar(cx, cy, 35, rr, "k", 0.8, 3.2 + i)
        fixation(cx, cy, 3.3 + i)
        # mouse pointer at the end of the line
        tx, ty = cx + rr * np.sin(np.deg2rad(35)), cy - rr * np.cos(np.deg2rad(35))
        cv.plot([tx + 0.6], [ty + 0.8], marker=(3, 0, 20), ms=3.0, color="white", mec="k", mew=0.3, zorder=3.4 + i)
    text(x + 0.2, y + SH + 0.5, lab, linespacing=1.0)

# payoff as a function of error: Exp. 1 (linear) and Exp. 2 (threshold at 8.5°), side by side
err = np.linspace(0, 45, 451)
payoff = {"Exp. 1": {GAIN: 1 - err / 45, LOSS: -err / 45},
          "Exp. 2": {GAIN: np.where(err <= 8.5, 1.0, 0.0), LOSS: np.where(err <= 8.5, 0.0, -1.0)}}
for k, (lab, f) in enumerate(payoff.items()):
    ax = axes_mm(143 + k * 19.0, yb0 + 2.5, 16, 13.5)
    ax.axhline(0, color=st.INC_COLORS[NEUTRAL], lw=st.LW_DATA, zorder=1)
    for inc in (GAIN, LOSS):
        ax.plot(err, f[inc], color=st.INC_COLORS[inc], lw=st.LW_DATA, zorder=2)
    ax.set_xticks([0, 45] if k == 0 else [8.5, 45], ["0", "45"] if k == 0 else ["8.5", "45"])
    ax.set_yticks([-1, 0, 1], ["−1", "0", "+1"] if k == 0 else [])
    ax.set_xlim(0, 45)
    ax.set_ylim(-1.15, 1.15)
    ax.set_title(lab, fontsize=st.SIZE_TEXT, pad=2)
    if k == 0:
        ax.set_ylabel("Payoff (£)", labelpad=1)
cv.text(143 + 16 + 1.5, yb0 + 20.7, "|Error| (°)", fontsize=st.SIZE_LABEL, ha="center", va="top")
SD.add("b", pd.DataFrame({"abs_error_deg": err, **{f"payoff_exp{lab[-1]}_{n}": f[inc] for lab, f in payoff.items()
                                                  for inc, n in ((GAIN, "gain"), (LOSS, "loss"))}}),
       "payoff (GBP) as a function of the absolute reproduction error, Experiments 1 and 2 (task design)")

# --------------------------------------------------------------------------- #
# c  Stimulus orientations
# --------------------------------------------------------------------------- #

yc0 = 33.7
letter(2, 0.5, yc0)
title(4.5, yc0 + 0.2, "Orientations")
ori = np.linspace(-90, 90, 361)
ax1 = axes_mm(8, yc0 + 5.6, 26, 17)
for mu, col, lab in ((-45, PINK, "B"), (45, BLUE, "A")):  # categories A (blue, +45°) and B (pink, −45°)
    dens = np.exp(0.5 * np.cos(np.deg2rad(2 * (ori - mu))))
    dens /= np.trapezoid(dens, ori)
    ax1.fill_between(ori, dens, color=col, alpha=0.12, lw=0)
    ax1.plot(ori, dens, color=col, lw=st.LW_DATA)
    ax1.text(mu, dens.max() * 1.04, lab, ha="center", va="bottom", fontsize=st.SIZE_MIN)
ax1.plot(ori, np.full_like(ori, 1 / 180), color="k", lw=st.LW_THIN, ls=(0, (2, 1)))
# the uniform density of the Stimulus Reproduction task: keyed at the bottom, below both category curves
ax1.legend(handles=[Line2D([], [], color="k", lw=st.LW_THIN, ls=(0, (2, 1)), label=st.TASK_REPRO)],
           loc="lower center", fontsize=st.SIZE_MIN, handlelength=1.6, borderaxespad=0.3)
ax1.set_xticks([-90, 0, 90], ["−90", "0", "90"])
ax1.set_xlabel("Orientation (°)", labelpad=1)
ax1.set_yticks([])
ax1.set_ylabel("Density", labelpad=2)
ax1.set_xlim(-90, 90)
ax1.set_ylim(0, None)
SD.add("c", pd.DataFrame({
    "orientation_deg": ori,
    **{f"density_category_{lab}": (lambda d: d / np.trapezoid(d, ori))(np.exp(0.5 * np.cos(np.deg2rad(2 * (ori - mu)))))
       for mu, lab in ((45, "A"), (-45, "B"))},
    "density_reproduction_uniform": np.full_like(ori, 1 / 180),
    "log_lr_A_vs_B": np.sin(np.deg2rad(2 * ori))}),
    "orientation densities (von Mises, kappa = 0.5) and evidence carried by one grating (task design)")
ax2 = axes_mm(46, yc0 + 5.6, 22, 17)
ax2.plot(ori, np.sin(np.deg2rad(2 * ori)), color="k", lw=st.LW_DATA)
ax2.axhline(0, color=st.REF_LINE, lw=st.LW_THIN, ls=(0, (3, 2)))
ax2.set_xticks([-90, 0, 90], ["−90", "0", "90"])
ax2.set_yticks([-1, 0, 1], ["−1", "0", "+1"])
ax2.set_xlim(-90, 90)
ax2.set_xlabel("Orientation (°)", labelpad=1)
ax2.set_ylabel("log LR\n(A : B)", labelpad=1, linespacing=1.0)

# --------------------------------------------------------------------------- #
# d  Integration trial, one screen per event
# --------------------------------------------------------------------------- #

xd0 = 76.0
letter(3, xd0 - 4, yc0)
title(xd0, yc0 + 0.2, f"{st.TASK_CATEG} task")
yd0 = yc0 + 4.3
seq = ["fix", "g", "g", "g", "dots", "g", "resp"]
thetas = iter((62, -18, 40, -55))
pos = {}
x, y = xd0, yd0
for i, kind in enumerate(seq):
    if kind == "dots":
        gap_ = 0.6 * DX  # extra horizontal step for the ellipsis
        cv.text(x + OV + (gap_ - OV) / 2, y + SH * 0.6, "···", fontsize=st.SIZE_LABEL, ha="center", va="center")
        x, y = x + gap_, y + DY * 0.6
        continue
    if kind == "resp":
        response_screen_integration(x, y, z=2 + i)
    else:
        cx, cy = screen(x, y, z=2 + i)
        image(color_wheel(), cx, cy, 0.66 * SH, z=3 + i)
        if kind == "fix":
            fixation(cx, cy, 3.3 + i)
        else:
            image(gabor(next(thetas), seed=20 + i, n=111), cx, cy, 0.56 * SH, z=3.1 + i)
    pos.setdefault(kind, []).append((x, y))
    x, y = x + DX, y + DY
(fx, fy), = pos["fix"]
g = pos["g"]
(rx, ry), = pos["resp"]
text(fx + 0.2, fy + SH + 0.5, "660 ms")
by = g[-1][1] + SH + 1.1
bx1 = g[-1][0] + DX - 0.6  # right end: where the response screen starts to overlap
cv.plot([g[0][0] + 0.5, g[0][0] + 0.5, bx1, bx1], [by - 0.7, by, by, by - 0.7], color="k", lw=st.LW_AXIS)
text((g[0][0] + bx1) / 2, by + 0.5, "4, 8 or 12 gratings: 100 ms each, every 333 ms (3 Hz)",
     ha="center")
text(rx + 0.2, ry + SH + 0.5, "until response\n(≤ 5 s, Exp. 2)", linespacing=1.0)
SD.note("d", "trial schematic (no data)")

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

df_m = pd.concat([pd.read_csv(ROOT / "results/prereg" / e / "behavior/data_mean_inc.csv").assign(
    participant=lambda d, e=e: d["participant"].astype(str) + "__" + e) for e in ("exp1", "exp2")], ignore_index=True)
df_m["incentive"] = df_m["incentive"].astype(float)


def wide(dv):
    d = df_m[["participant", "incentive", dv]].dropna()
    return d, d.pivot(index="participant", columns="incentive", values=dv).dropna()


yr, hr = 72.3, 28.0  # results row: top and axes height (mm)

# e  Estimation
letter(4, 0.5, yr - 5)
title(4.5, yr - 4.8, st.TASK_REPRO)
d, w = wide("estim_sd")
SD.add_levels("e", w, "estim_sd", "Stimulus Reproduction error SD (deg) per participant and condition, and its "
              "change from neutral")
ax = axes_mm(10, yr, 26, hr)
dist.levels_panel(ax, w, ylabel="Error SD (°)")
ax = axes_mm(46, yr, 16, hr)
dist.effects_panel(ax, dist.incentive_effects(w), ylabel="Δ error SD (°)",
                   pvals=paper_holm_p(ROOT, "estim_sd"))

# f  Integration: choices as a function of evidence
letter(5, 66, yr - 5)
title(70, yr - 4.8, st.TASK_CATEG)
# binned data and the mean of the participants' probit fits (behavior_stats.psychometric_consistency)
BX = ROOT / "results/exploratory/merged/behavior/stats/behav_extra/params"
bins = pd.read_csv(BX / "psychometric_consistency_bins.csv")
curves = pd.read_csv(BX / "psychometric_consistency_curves.csv")
SD.add("f", bins, "proportion of choices consistent with the evidence per bin of |summed evidence|: mean and "
       "s.e.m. over participants", part="bins")
SD.add("f", curves, "mean of the participants' probit fits", part="curves")
ax = axes_mm(78, yr, 28, hr)
for inc in (NEUTRAL, LOSS, GAIN):
    g, c = bins[bins["incentive"] == inc], curves[curves["incentive"] == inc]
    col = st.INC_COLORS[inc]
    ax.plot(c["abs_evidence"].to_numpy(), c["p_consistent"].to_numpy(), color=col, lw=st.LW_DATA,
            zorder=3 if inc != NEUTRAL else 2)
    ax.errorbar(g["abs_evidence"].to_numpy(), g["p_consistent_mean"].to_numpy(), yerr=g["p_consistent_sem"].to_numpy(),
                fmt="o", ms=2.2, color=col, mec="none", elinewidth=st.LW_AXIS, zorder=4)
ax.set_xlim(0, 5.5)
ax.set_ylim(0.45, 1.0)
ax.set_yticks([0.5, 0.75, 1.0])
ax.set_xlabel("|Evidence| (Σ log LR)")
ax.set_ylabel("P(choice consistent\nwith evidence)")
ax.axhline(0.5, color=st.REF_LINE, lw=st.LW_THIN, ls=(0, (3, 2)), zorder=0)
handles = [plt.Line2D([], [], color=st.INC_COLORS[i], marker="o", ms=2.2, mec="none", lw=st.LW_DATA,
                      label=st.INC_TICKS[i]) for i in (LOSS, NEUTRAL, GAIN)]
ax.legend(handles=handles, title=st.INC_XLABEL, loc="lower right", bbox_to_anchor=(1.0, 0.12), handlelength=1.4)

# g  Integration: sensitivity
letter(6, 110, yr - 5)
d, w = wide("sensitivity")
SD.add_levels("g", w, "sensitivity", "evidence sensitivity (probit slope) per participant and condition, and its "
              "change from neutral")
ax = axes_mm(120, yr, 24, hr)
dist.levels_panel(ax, w, ylabel="Evidence sensitivity")
ax = axes_mm(155, yr, 16, hr)
dist.effects_panel(ax, dist.incentive_effects(w), ylabel="Δ sensitivity",
                   pvals=paper_holm_p(ROOT, "sensitivity"))

st.save(fig, OUT / "figures" / "fig1")
SD.write()
print("saved", OUT / "figures" / "fig1.svg")
