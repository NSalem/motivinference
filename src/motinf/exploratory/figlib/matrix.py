"""Matrices at print scale (Supplementary Figures 2, 4 and 5): confusion matrices of model recovery and
correlation matrices of parameter recovery. Cell text is 6 pt, black or white by cell luminance; no colour bar
when every cell carries its value."""

from __future__ import annotations

import numpy as np

from motinf.exploratory.figlib import figstyle as st


def ink(rgba) -> str:
    """Black or white text for a cell of this colour."""
    r, g, b = rgba[:3]
    return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else "black"


def matrix(ax, M, *, cmap, vmin, vmax, xlabels, ylabels=None, fmt=lambda i, j, v: f"{v:.0f}", xlabel=None,
           ylabel=None, fontsize=st.SIZE_TEXT, grid="white", grid_lw=0.6):
    """Draw ``M`` (rows: generative, columns: recovered) with one value per cell."""
    M = np.asarray(M, float)
    n, m = M.shape
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal", interpolation="nearest")
    for i in range(n):
        for j in range(m):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt(i, j, v), ha="center", va="center", fontsize=fontsize, linespacing=1.0,
                        color=ink(im.cmap(im.norm(v))))
    ax.set_xticks(range(m), xlabels)
    ax.set_yticks(range(n), ylabels if ylabels is not None else xlabels)
    ax.tick_params(length=0, pad=1.5)
    for s in ax.spines.values():
        s.set_visible(False)
    for k in range(1, n):
        ax.axhline(k - 0.5, color=grid, lw=grid_lw)
    for k in range(1, m):
        ax.axvline(k - 0.5, color=grid, lw=grid_lw)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    return im
