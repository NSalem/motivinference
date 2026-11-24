"""Figure 4: repeated trials separate bias from noise.

Panels:
    a  repeated trials: the correlation ρ between the two presentations of a stimulus or sequence splits the
       variance into bias (ρσ²) and noise ((1 − ρ)σ²) (simulated, with ρ set to the observed means at £0)
    b  Stimulus Reproduction: noise SD and bias SD per condition and their change from £0 (log scale), and the
       noise and bias variances as a share of the total at £0 (95% bootstrap CIs)
    c  Sequence Categorization: the same, in log LR units

Inputs: results/exploratory/exp{1,2}/variance_partition/{variance_partition_estim,var_partition_categ}.csv
(pooled); results/exploratory/merged/paper_tables/paper_tables.json (stars: Table 3).
Output: plots/exploratory/figures/fig4.{svg,pdf,png}, at print size (180 mm wide).

Usage (from the repository root):
    python scripts/exploratory/figures/fig4.py
"""

from __future__ import annotations

import os

import figpaths  # noqa: F401  (paths first)

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, Patch, Polygon, Rectangle

from motinf.exploratory import figure_data as V  # Fig. 4 computations
from motinf.exploratory.figlib import dist, figstyle as st
from motinf.exploratory.figlib.canvas import Canvas, gabor
from motinf.exploratory.figlib.source_data import SourceData

st.apply()
ROOT, OUT = figpaths.DATA_ROOT, figpaths.OUT_ROOT
SD = SourceData("fig4", OUT)
W, H = st.LAYOUT_DOUBLE, 143.0  # layout mm; saved at st.DOUBLE wide (st.page_scale)
LOSS, NEUTRAL, GAIN = st.LOSS, st.NEUTRAL, st.GAIN
INC = (LOSS, NEUTRAL, GAIN)
DASH = (0, (3, 1.5))

c = Canvas(W, H)
cv = c.cv

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

base = ROOT / "results/exploratory"
df_e = pd.concat([V.tag_participant_col(pd.read_csv(base / e / "variance_partition/variance_partition_estim.csv"), e)
                  for e in ("exp1", "exp2")], ignore_index=True)
df_c = pd.concat([V.tag_participant_col(pd.read_csv(base / e / "variance_partition/var_partition_categ.csv"), e)
                  for e in ("exp1", "exp2")], ignore_index=True)
df_e_deg = df_e.copy()
for col in ("noise_sd", "bias_sd", "total_sd"):  # Estimation SDs in degrees, as in Table 3
    df_e_deg[col] = np.rad2deg(df_e_deg[col].to_numpy(float))
RHO = {"estim": V.wide_by_incentive(df_e, "rho")[0.0].mean(), "categ": V.wide_by_incentive(df_c, "rho")[0.0].mean()}  # £0 means
SD.note("a", "schematic; the scatter plots are simulated with rho set to the mean at neutral "
        f"(Stimulus Reproduction {RHO['estim']:.3f}, Sequence Categorization {RHO['categ']:.3f})")

# --------------------------------------------------------------------------- #
# a  Repeated trials
# --------------------------------------------------------------------------- #

c.letter(0, 0.5, 1.6)
c.title(4.5, 1.8, "Repeated trials separate bias from noise")
DYA = 0.4  # panel a below its title
rng = np.random.default_rng(7)
PINK, BLUE = "#e377c2", "#4a90d9"  # category colours of the Integration task (as in Figs 1 and 2)
CONS_DARK, CONS_LIGHT = "#1f6e3d", "#9fd3ad"  # error consistency: consistent (dark) to inconsistent (light)
INCONS = "0.55"  # inconsistent choices
CMAP_CONS = LinearSegmentedColormap.from_list("cons", [CONS_LIGHT, CONS_DARK])
XS = (14.0, 30.0)  # the two presentations of the stimulus
XD = (47.0, 59.0)  # decision variable at the two presentations (x1, x2)
XL = 67.0  # case labels
XSC, WSC = 102.0, 22.0  # scatter plots
ROWS = {"estim": (11.5 + DYA, 34.0 + DYA), "categ": (39.0 + DYA, 61.5 + DYA)}  # top and bottom of each row (mm)


def brace(x, y0, y1, w=1.6, lw=st.LW_AXIS):
    """Left curly brace from y0 to y1 (mm), opening to the right."""
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    ym, r, xm = (y0 + y1) / 2, w / 2, x + w / 2
    verts = [(x + w, y0), (xm, y0), (xm, y0 + r), (xm, ym - r), (xm, ym), (x, ym), (xm, ym), (xm, ym + r),
             (xm, y1 - r), (xm, y1), (x + w, y1)]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.CURVE3,
             Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3]
    cv.add_patch(PathPatch(Path(verts, codes), fc="none", ec="k", lw=lw, zorder=3))


def gradient_bar(x0, y0, w, h, c_from, c_to, horizontal=True, alpha=None, z=3):
    """Colour gradient in a rectangle (mm): from c_from (left / bottom) to c_to (right / top)."""
    n = 64
    t = np.linspace(0, 1, n)
    a, b = np.array(to_rgb(c_from)), np.array(to_rgb(c_to))
    rgb = (1 - t)[:, None] * a + t[:, None] * b
    img = rgb[None, :, :] if horizontal else rgb[::-1][:, None, :]
    if alpha is not None:
        al = alpha[None, :, None] if horizontal else alpha[::-1][:, None, None]
        img = np.concatenate([np.broadcast_to(img, (img.shape[0], img.shape[1], 3)),
                              np.broadcast_to(al, (img.shape[0], img.shape[1], 1))], axis=2)
    cv.imshow(img, extent=(x0, x0 + w, y0 + h, y0), aspect="auto", interpolation="bilinear", zorder=z)


def evidence_arrow(cx, cy, L, *, dv=None, truth=None, choice=None, h=1.5, z=3):
    """Pink-to-blue double arrow (length L mm) for the summed evidence; ``dv``: black tick at the decision
    variable (in units of the half-length); ``truth``: dotted tick at the presented evidence; ``choice``: the
    chosen side ("blue" / "pink") at full colour, the other faded."""
    x0, x1 = cx - L / 2, cx + L / 2
    hl = 1.4  # arrowhead length
    alpha = np.ones(64)
    if choice == "blue":
        alpha = np.where(np.linspace(0, 1, 64) < 0.5, 0.3, 1.0)
    elif choice == "pink":
        alpha = np.where(np.linspace(0, 1, 64) < 0.5, 1.0, 0.3)
    gradient_bar(x0 + hl, cy - h / 2, L - 2 * hl, h, PINK, BLUE, alpha=alpha, z=z)
    for side, col in ((-1, PINK), (1, BLUE)):
        faded = (choice == "blue" and side < 0) or (choice == "pink" and side > 0)
        xb = cx + side * (L / 2 - hl)
        tip = cx + side * L / 2
        cv.add_patch(Polygon([(xb, cy - 1.25 * h), (tip, cy), (xb, cy + 1.25 * h)], closed=True, fc=col,
                             ec="none", alpha=0.3 if faded else 1.0, zorder=z))
    half = L / 2 - hl
    if truth is not None:
        cv.plot([cx + truth * half] * 2, [cy - h, cy + h], color="k", lw=0.5, ls=(0, (1, 1)), zorder=z + 0.2)
    if dv is not None:
        cv.plot([cx + dv * half] * 2, [cy - 1.1 * h, cy + 1.1 * h], color="k", lw=0.9, zorder=z + 0.3)


def width_arrow(ax, centre, half, label):
    """Double arrow across the ellipse (perpendicular to the diagonal), labelled; data coordinates."""
    d = np.array([-1.0, 1.0]) / np.sqrt(2)
    p0, p1 = np.array(centre) - half * d, np.array(centre) + half * d
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="<|-|>,head_length=0.35,head_width=0.18",
                                                     color="0.3", lw=0.6, shrinkA=0, shrinkB=0), zorder=6)
    ax.text(p1[0] + 0.02 * np.ptp(ax.get_xlim()), p1[1], label, ha="left", va="center", fontsize=st.SIZE_TEXT)


# headers and row labels
c.text(sum(XS) / 2, 5.0 + DYA, "Repeated stimulus", ha="center")
c.text(sum(XD) / 2, 5.0 + DYA, "Decision variable", ha="center")
for x, lab in zip(XD, (r"$x_1$", r"$x_2$")):
    c.text(x, 9.6 + DYA, lab, ha="center", va="center")
for k, lab in (("estim", st.TASK_REPRO), ("categ", st.TASK_CATEG)):
    y0, y1 = ROWS[k]
    c.text(2.9, (y0 + y1) / 2, lab.replace(" ", "\n"), rotation=90, ha="center", va="center", fontweight="bold",
           fontsize=st.SIZE_LABEL, linespacing=1.0)
    brace(37.0, y0, y1)

# -- Estimation ---------------------------------------------------------------------------------------
y0, y1 = ROWS["estim"]
ys = np.linspace(y0 + 3.4, y1 - 3.4, 3)  # three example cases
TH = 30.0
for x, lab in zip(XS, (r"$s_1$", r"$s_2$")):
    c.image(gabor(TH, seed=5, n=111), x, (y0 + y1) / 2 + 1.5, 9.5, z=3)
    c.orient_line(x, (y0 + y1) / 2 + 1.5, TH, 5.6, color="0.25", lw=0.5, ls=(0, (2, 1.2)), zorder=4)
    c.text(x, (y0 + y1) / 2 - 4.4, lab, ha="center", va="bottom")
c.text(sum(XS) / 2, (y0 + y1) / 2 + 1.5, "=", ha="center", va="center", fontsize=st.SIZE_LABEL)
cases_e = (("accurate", (0.0, 0.0), "0.55"), ("consistent\nerrors", (24.0, 20.0), CONS_DARK),
           ("inconsistent\nerrors", (22.0, -26.0), CMAP_CONS(0.3)))
for yv, (lab, errs, col) in zip(ys, cases_e):
    for x, e in zip(XD, errs):
        cv.add_patch(Circle((x, yv), 3.2, fc="white", ec="0.6", lw=st.LW_THIN, zorder=3))
        c.orient_line(x, yv, TH, 3.2, color="0.45", lw=0.5, ls=(0, (2, 1.2)), zorder=3.2)
        c.orient_line(x, yv, TH + e, 3.2, color=col, lw=1.1, solid_capstyle="round", zorder=3.4)
    c.text(XL, yv, lab, va="center", linespacing=1.0)
# colour key: error consistency
gradient_bar(83.0, ys[0] - 2.0, 1.6, ys[-1] - ys[0] + 4.0, CONS_LIGHT, CONS_DARK, horizontal=False)
cv.add_patch(Polygon([(82.4, ys[0] - 2.0), (83.8, ys[0] - 3.6), (85.2, ys[0] - 2.0)], closed=True, fc=CONS_DARK,
                     ec="none", zorder=3))
c.text(87.0, (ys[0] + ys[-1]) / 2, "error\nconsistency", rotation=90, ha="center", va="center", linespacing=1.0)
# scatter of the errors, coloured by their consistency
r_e, sd_e = RHO["estim"], 8.0
pts = rng.multivariate_normal([0, 0], sd_e ** 2 * np.array([[1, r_e], [r_e, 1]]), size=90)
cons = 1 - np.clip(np.abs(pts[:, 0] - pts[:, 1]) / 22.0, 0, 1)
ax = c.axes_mm(XSC, y0, WSC, y1 - y0 - 2.0)
lim = 27
ax.axhline(0, color=st.REF_LINE, lw=st.LW_THIN, zorder=1)
ax.axvline(0, color=st.REF_LINE, lw=st.LW_THIN, zorder=1)
ax.scatter(pts[:, 0], pts[:, 1], s=2.4, c=CMAP_CONS(cons), ec="none", zorder=3)
ax.add_patch(Ellipse((0, 0), 2 * 2 * sd_e * np.sqrt(1 + r_e), 2 * 2 * sd_e * np.sqrt(1 - r_e), angle=45,
                     fill=False, ec="0.3", lw=st.LW_DATA, zorder=4))
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_xticks([-20, 0, 20], ["−20", "0", "20"])
ax.set_yticks([-20, 0, 20], ["−20", "0", "20"])
ax.set_xlabel(r"Error $x_1$ (°)", labelpad=1)
ax.set_ylabel(r"Error $x_2$ (°)", labelpad=1)
width_arrow(ax, (14.5, 14.5), 2 * sd_e * np.sqrt(1 - r_e) * 0.9, r"$1-\rho$")

# -- Integration --------------------------------------------------------------------------------------
y0, y1 = ROWS["categ"]
ys = np.linspace(y0 + 3.0, y1 - 3.0, 3)
thetas = (40, -25, 60)
yseq = y0 + 7.2
for x, lab in zip(XS, (r"$s_1$", r"$s_2$")):
    for j, th in enumerate(thetas):
        gx = x - 3.4 + 3.4 * j
        c.image(gabor(th, seed=11 + j, n=91), gx, yseq, 6.4, z=3 + 0.1 * j)
        cv.add_patch(Circle((gx, yseq), 3.2, fc="none", ec=BLUE if np.sin(np.deg2rad(2 * th)) > 0 else PINK,
                            lw=0.8, zorder=3.05 + 0.1 * j))
    c.text(x, yseq - 4.6, lab, ha="center", va="bottom")
    evidence_arrow(x, yseq + 7.6, 12.0, truth=0.35)
c.text(sum(XS) / 2, yseq, "=", ha="center", va="center", fontsize=st.SIZE_LABEL)
c.text(sum(XS) / 2, yseq + 7.6, "=", ha="center", va="center", fontsize=st.SIZE_LABEL)
c.text(sum(XS) / 2, yseq + 10.2, "summed evidence", ha="center", va="top")
cases_c = (("accurate", (0.6, 0.3), BLUE), ("consistent\nerrors", (-0.45, -0.6), st.tint(CONS_DARK, 0.55)),
           ("inconsistent\nerrors", (0.55, -0.45), INCONS))
for yv, (lab, dvs, sw) in zip(ys, cases_c):
    for x, dv in zip(XD, dvs):
        evidence_arrow(x, yv, 10.5, dv=dv, truth=0.35, choice="blue" if dv > 0 else "pink")
    c.rbox(XL - 2.9, yv - 1.0, 2.0, 2.0, fc=sw, r=0.2, z=3)
    c.text(XL, yv, lab, va="center", linespacing=1.0)
# scatter of the decision variables at the two presentations: same / different choices
r_c, mu_c = RHO["categ"], 0.9
ptc = rng.multivariate_normal([mu_c, mu_c], np.array([[1, r_c], [r_c, 1]]), size=90)
ax = c.axes_mm(XSC, y0, WSC, y1 - y0 - 2.0)
LO_C, HI_C = -2.6, 4.4
ax.add_patch(Rectangle((LO_C, LO_C), -LO_C, -LO_C, fc=st.tint(CONS_DARK, 0.25), ec="none", zorder=0))
for (x0q, y0q, wq, hq) in ((0, LO_C, HI_C, -LO_C), (LO_C, 0, -LO_C, HI_C)):
    ax.add_patch(Rectangle((x0q, y0q), wq, hq, fc="0.9", ec="none", zorder=0))
ax.axhline(0, color=st.REF_LINE, lw=st.LW_THIN, zorder=1)
ax.axvline(0, color=st.REF_LINE, lw=st.LW_THIN, zorder=1)
both_blue = (ptc[:, 0] > 0) & (ptc[:, 1] > 0)
both_pink = (ptc[:, 0] < 0) & (ptc[:, 1] < 0)
cols = np.where(both_blue, BLUE, np.where(both_pink, CONS_DARK, INCONS))
ax.scatter(ptc[:, 0], ptc[:, 1], s=2.4, c=list(cols), ec="none", zorder=3)
ax.add_patch(Ellipse((mu_c, mu_c), 2 * 2 * np.sqrt(1 + r_c), 2 * 2 * np.sqrt(1 - r_c), angle=45, fill=False,
                     ec="0.3", lw=st.LW_DATA, zorder=4))
ax.set_xlim(LO_C, HI_C)
ax.set_ylim(LO_C, HI_C)
ax.set_xticks([0])
ax.set_yticks([0])
for side in ("left", "bottom"):
    ax.spines[side].set_visible(False)
ax.tick_params(length=0, pad=4.0)
width_arrow(ax, (mu_c + 1.6, mu_c + 1.6), 2 * np.sqrt(1 - r_c) * 0.9, r"$1-\rho$")
# the axes as pink-to-blue arrows, as for the decision variable
ya_ = y1 - 2.0 + 0.9
evidence_arrow(XSC + WSC / 2, ya_, WSC, h=1.0)
xa_ = XSC - 0.9
gradient_bar(xa_ - 0.5, y0 + 1.2, 1.0, (y1 - 2.0 - y0) - 2.4, PINK, BLUE, horizontal=False)
for sgn, col, yy in ((1, BLUE, y0), (-1, PINK, y1 - 2.0)):
    cv.add_patch(Polygon([(xa_ - 1.25, yy + sgn * 1.2), (xa_, yy), (xa_ + 1.25, yy + sgn * 1.2)], closed=True,
                         fc=col, ec="none", zorder=3))
c.text(XSC + WSC + 0.8, ya_, r"$x_1$", va="center")  # axis names at the arrow tips
c.text(xa_ - 1.6, y0 + 0.6, r"$x_2$", ha="right", va="center")

# partition of the variance
xq = 136.0
c.text(xq, 17.0 + DYA, "Partition", fontweight="bold", fontsize=st.SIZE_LABEL)
for j, eq in enumerate((r"$\sigma^2 = \sigma^2_{\mathrm{bias}} + \sigma^2_{\mathrm{noise}}$",
                        r"$\sigma^2_{\mathrm{bias}} = \rho\,\sigma^2$",
                        r"$\sigma^2_{\mathrm{noise}} = (1 - \rho)\,\sigma^2$")):
    c.text(xq, 25.5 + DYA + 6.5 * j, eq, va="center", fontsize=st.SIZE_LABEL)
c.text(xq, 42.0 + DYA, r"$\rho$: correlation between $x_1$ and $x_2$" + "\n" + f"({st.TASK_REPRO}: the errors;"
       + "\n" + f"{st.TASK_CATEG}: estimated" + "\n" + "from choice consistency)", va="top", linespacing=1.15)

# --------------------------------------------------------------------------- #
# b, c  Noise and bias per task
# --------------------------------------------------------------------------- #

GX = {"noise": 0.0, "bias": 3.6}  # x offset of each group in the per-condition panel
EX = {"noise": 0.0, "bias": 2.5}  # in the change panel


def group_title(ax, xs, s, dy_pt=0.0):
    from matplotlib.transforms import blended_transform_factory, offset_copy
    tr = offset_copy(blended_transform_factory(ax.transData, ax.transAxes), fig=c.fig, y=dy_pt, units="points")
    ax.text(np.mean(xs), 1.0, s, transform=tr, ha="center", va="bottom", fontsize=st.SIZE_TEXT, fontweight="bold")


def task_row(i_letter, title, yt, df_sd, df_raw, sd_label, task):
    c.letter(i_letter, 0.5, yt)
    c.title(4.5, yt + 0.2, title)
    panel = st.letter(i_letter)
    for name in ("noise", "bias"):
        SD.add_levels(panel, V.wide_by_incentive(df_sd, f"{name}_sd"), f"{name}_sd",
                      f"{title}: {name} SD {sd_label.split(' ', 1)[1]} per participant and condition, and the change "
                      "of its log from neutral", change="logdiff", part=name)
    yax, hax = yt + 9.0, 22.0
    # per condition: noise | bias
    ax = c.axes_mm(12.0, yax, 50.0, hax)
    allv = []
    for name in ("noise", "bias"):
        w = V.wide_by_incentive(df_sd, f"{name}_sd")
        xs = dist.levels_panel(ax, w, x0=GX[name], fill=(name == "noise"), finish=False)
        group_title(ax, xs, name.capitalize(), dy_pt=8.0)
        allv += [w[k].to_numpy(float) for k in INC]
    lo, hi = dist.robust_limits(allv, k=3.0, include=(0.0,))
    ax.set_ylim(0, hi)
    for name in ("noise", "bias"):
        w = V.wide_by_incentive(df_sd, f"{name}_sd")
        for j, k in enumerate(INC):
            dist.mark_out_of_range(ax, GX[name] + j + 0.12 + 0.55 * 0.5, w[k].to_numpy(float), 0, hi)
    ticks = [GX[n] + j for n in ("noise", "bias") for j in range(3)]
    ax.set_xticks(ticks, [st.INC_TICKS[k] for _ in range(2) for k in INC])
    ax.set_xlim(-0.35, GX["bias"] + 2 + 0.12 + 0.55 + 0.05)
    ax.set_xlabel(st.INC_XLABEL)
    ax.set_ylabel(sd_label)
    # change from £0 (log scale): noise | bias
    ax = c.axes_mm(76.0, yax, 34.0, hax)
    eff = {n: dist.incentive_effects(np.log(V.wide_by_incentive(df_sd, f"{n}_sd"))) for n in ("noise", "bias")}
    lim = dist.robust_limits([v for n in eff for v in eff[n].values()], k=3.0)
    for name in ("noise", "bias"):
        p = V.paper_holm_p(ROOT, f"{task}_noise_log" if name == "noise" else f"{task}_bias")  # Table 3
        xs = dist.effects_panel(ax, eff[name], x0=EX[name], fill=(name == "noise"), ylim=lim,
                                pvals=p, finish=False)
        group_title(ax, xs, name.capitalize(), dy_pt=8.0)
    dist.zero_line(ax)
    ax.set_xticks([EX[n] + j for n in ("noise", "bias") for j in range(2)],
                  [st.INC_TICKS[k] for _ in range(2) for k in (LOSS, GAIN)])
    ax.set_xlim(-0.35, EX["bias"] + 1 + 0.12 + 0.55 + 0.05)
    ax.set_xlabel(st.EFF_XLABEL)
    ax.set_ylabel("Δ log SD")
    dist.pct_axis(ax, "change")
    # noise and bias variance as a share of the £0 total
    s = V.prop_of_baseline_summary(df_raw).set_index("incentive")
    SD.add(panel, s.reset_index(), f"{title}: noise and bias variance per condition as a share of the total at "
           "neutral (mean, 95% bootstrap CI)", part="variance_shares")
    ax = c.axes_mm(134.0, yax, 45.0, hax)
    HB = 0.62
    for row, k in enumerate(INC):
        col = st.INC_COLORS[k]
        r = s.loc[int(k)]
        nm, bm = float(r["noise_m"]), float(r["bias_m"])
        ax.barh(row, nm, height=HB, color=st.tint(col, 0.45), ec=col, lw=0.6, zorder=2)
        ax.barh(row, bm, left=nm, height=HB, color="white", ec=col, lw=0.6, ls=DASH, zorder=2)
        ax.errorbar(nm, row, xerr=[[nm - r["noise_lo"]], [r["noise_hi"] - nm]], fmt="none", color="0.2",
                    lw=st.LW_THIN, capsize=1.0, capthick=st.LW_THIN, zorder=5)
        ax.errorbar(nm + bm, row, xerr=[[bm - r["bias_lo"]], [r["bias_hi"] - bm]], fmt="none", color="0.2",
                    lw=st.LW_THIN, capsize=1.0, capthick=st.LW_THIN, zorder=5)
        ax.text(nm / 2, row + 0.06, f"{round(nm * 100):d}%", ha="center", va="center", fontsize=st.SIZE_TEXT, zorder=6)
        ax.text(nm + bm / 2, row + 0.06, f"{round(bm * 100):d}%", ha="center", va="center", fontsize=st.SIZE_TEXT,
                zorder=6)
    n0 = float(s.loc[0, "noise_m"])
    ax.axvline(1.0, color=st.REF_LINE, lw=st.LW_THIN, ls=(0, (3, 2)), zorder=1)
    ax.axvline(n0, color="k", lw=st.LW_THIN, zorder=1)
    ax.set_yticks(range(3), [st.INC_TICKS[k] for k in INC])
    ax.set_ylim(2.55, -0.55)
    ax.set_xlim(0, 1.08)
    ax.set_xticks([0, 0.5, 1.0], ["0", "50", "100"])
    ax.spines["bottom"].set_bounds(0, 1)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_ylabel(st.INC_XLABEL)
    ax.set_xlabel("Variance (% of the total at £0)")
    return ax


ax_b = task_row(1, st.TASK_REPRO, 65.3, df_e_deg, df_e, "SD (°)", "estim")
task_row(2, st.TASK_CATEG, 104.6, df_c, df_c, "SD (log LR)", "categ")
# key for the filled / outlined convention, once, above the first stacked-bar panel
ax_b.legend(handles=[Patch(fc=st.tint("0.5", 0.45), ec="0.5", lw=0.6, label="noise"),
                     Line2D([], [], color="k", lw=st.LW_THIN, label="£0 noise"),
                     Patch(fc="white", ec="0.5", lw=0.6, ls=DASH, label="bias"),
                     Line2D([], [], color=st.REF_LINE, lw=st.LW_THIN, ls=(0, (3, 2)), label="£0 total")],
            loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, handlelength=1.4, columnspacing=1.0,
            borderaxespad=0.2, labelspacing=0.2)

if os.environ.get("FIG_DEBUG"):
    c.fig.canvas.draw()
    rr = c.fig.canvas.get_renderer()
    k_ = 25.4 / c.fig.dpi
    for a in c.fig.axes[1:]:
        p_, t_ = a.get_window_extent(rr), a.get_tightbbox(rr)
        print(f"axes {p_.x0 * k_:6.1f}-{p_.x1 * k_:6.1f} | labels {t_.x0 * k_:6.1f}-{t_.x1 * k_:6.1f} "
              f"| y {H - p_.y1 * k_:5.1f}-{H - p_.y0 * k_:5.1f} | labels y {H - t_.y1 * k_:5.1f}-{H - t_.y0 * k_:5.1f}")
st.save(c.fig, OUT / "figures" / "fig4")
SD.write()
print("saved", OUT / "figures" / "fig4.svg", "| rho £0:", {k: round(v, 3) for k, v in RHO.items()},
      "| N:", df_e["participant"].nunique(), df_c["participant"].nunique())
