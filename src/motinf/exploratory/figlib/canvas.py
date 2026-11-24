"""Whole-figure canvas in millimetres (origin top-left, y pointing down) and schematic helpers.

A figure is created at its final size; schematics and text are drawn on one full-figure axes in mm,
data panels are placed with ``axes_mm``. Stimulus helpers render the task's gratings in numpy.
Used by the figure scripts in ``scripts/exploratory/figures/``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from motinf.exploratory.figlib import figstyle as st


class Canvas:
    def __init__(self, w: float, h: float):
        """``w`` × ``h``: the layout in mm (``st.LAYOUT_DOUBLE`` wide for double-column figures). The figure is
        created at that size × ``st.page_scale(w)``; every mm coordinate below is in layout mm."""
        self.W, self.H = w, h
        self.scale = st.page_scale(w)
        self.fig = plt.figure(figsize=st.mm(w * self.scale, h * self.scale))
        cv = self.fig.add_axes([0, 0, 1, 1])
        cv.set_xlim(0, w)
        cv.set_ylim(h, 0)
        cv.axis("off")
        cv.set_zorder(0)
        self.cv = cv

    def axes_mm(self, x, y, w, h, z=2, bg=True):
        """Data axes at (x, y) mm from the top-left corner, w × h mm. ``bg=False``: no axes background, so a
        shaded panel background (``box``) shows through."""
        ax = self.fig.add_axes([x / self.W, 1 - (y + h) / self.H, w / self.W, h / self.H])
        ax.set_zorder(z)
        ax.patch.set_visible(bg)
        return ax

    def text(self, x, y, s, **kw):
        kw = {"fontsize": st.SIZE_TEXT, "ha": "left", "va": "top", **kw}
        return self.cv.text(x, y, s, **kw)

    def letter(self, i, x, y):
        self.cv.text(x, y, st.letter(i), fontsize=st.SIZE_LETTER, fontweight="bold", ha="left", va="top")

    def title(self, x, y, s):
        self.cv.text(x, y, s, fontsize=st.SIZE_LABEL, fontweight="bold", ha="left", va="top")

    def rbox(self, x, y, w, h, *, fc, ec="none", lw=st.LW_AXIS, r=0.6, z=3, ls="-", alpha=1.0):
        self.cv.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec,
                                         lw=lw, zorder=z, ls=ls, alpha=alpha))

    def box(self, x, y, w, h, *, fc, z=0.5):
        """Plain rectangle (square corners, no edge), for shaded panel backgrounds: drawn under everything else."""
        self.cv.add_patch(Rectangle((x, y), w, h, fc=fc, ec="none", lw=0, zorder=z))

    def arrow(self, x0, y0, x1, y1, *, lw=st.LW_AXIS, color="k", head=3.0, z=4, style="-|>"):
        """Straight arrow in mm; ``head``: head length in points."""
        self.cv.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=f"{style},head_length=1,head_width=0.5",
                                          mutation_scale=head, lw=lw, color=color, zorder=z, shrinkA=0,
                                          shrinkB=0))

    def image(self, img, cx, cy, d, z=3):
        self.cv.imshow(img, extent=(cx - d / 2, cx + d / 2, cy + d / 2, cy - d / 2), zorder=z,
                       interpolation="bilinear")

    def orient_line(self, cx, cy, th, r, **kw):
        """Line through (cx, cy) mm at orientation ``th`` (degrees clockwise from vertical), half-length ``r`` mm."""
        t = np.deg2rad(th)
        self.cv.plot([cx - r * np.sin(t), cx + r * np.sin(t)], [cy + r * np.cos(t), cy - r * np.cos(t)], **kw)


def gabor(theta_deg, n=121, cycles=3.2, contrast=0.4, noise=0.1, seed=0):
    """Grating in a circular aperture (RGBA, transparent outside); theta clockwise from vertical."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]
    th = np.deg2rad(theta_deg)
    u = x * np.cos(th) + y * np.sin(th)  # stripes run along theta (image rows point down)
    g = 0.5 + 0.5 * contrast * np.cos(2 * np.pi * cycles * u / 2 + rng.uniform(0, 2 * np.pi))
    g = g + noise * 0.5 * rng.standard_normal(g.shape)
    r = np.hypot(x, y)
    return np.clip(np.dstack([g, g, g, (r <= 1).astype(float)]), 0, 1)
