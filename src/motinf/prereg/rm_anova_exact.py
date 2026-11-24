"""Repeated-measures ANOVA with the *exact* Greenhouse-Geisser epsilon of every effect.

Why this exists: ``pingouin.rm_anova`` estimates epsilon for a two-way design from the covariance
matrix of *all* cells treated as one factor. That is right for main effects but wrong for an
interaction when both factors have more than two levels (pingouin warns about this). Example, 3 x 3
log choice variability, Experiment 1: pingouin epsilon(interaction) = 0.43, exact = 0.72.

Method (Greenhouse & Geisser, 1959; Box, 1954). For one effect, take an orthonormal contrast matrix
``C`` (m x cells) spanning the effect, project each subject's cell means, ``Z = Y C'``, and let
``S = cov(Z)``. Then ``F = n * sum(mean(Z)^2) / tr(S)`` (identical to the usual univariate F), and
``eps = tr(S)^2 / (m * tr(S^2))``. The GG p-value uses ``F`` with ``(m*eps, m*(n-1)*eps)`` df.
Epsilon is invariant to the choice of orthonormal basis, so any orthonormal ``C`` gives the same value.

Main effects and one-way designs reproduce pingouin exactly; only interactions differ.

Not part of the code frozen at pre-registration: added to the pre-registered tier for deviation 2 of
``PREREG_DEVIATIONS.md`` (every repeated-measures ANOVA reports the GG-corrected p with the exact epsilon).
The pre-registered scripts report it next to pingouin's values; the exploratory analyses import it from here.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


def helmert(k: int) -> np.ndarray:
    """Orthonormal contrasts (rows) of a k-level factor, shape (k-1, k); each row sums to 0."""
    h = np.zeros((k - 1, k))
    for i in range(1, k):
        h[i - 1, :i] = 1.0
        h[i - 1, i] = -float(i)
    return h / np.linalg.norm(h, axis=1, keepdims=True)


def effect_contrasts(n_levels: list[int], subset: tuple[int, ...]) -> np.ndarray:
    """Contrast matrix of the effect involving the factors in ``subset`` (indices into n_levels).

    Cells are assumed ordered factor-major (first factor slowest), as produced by sorting a
    ``pivot_table`` on a column MultiIndex.
    """
    c = np.ones((1, 1))
    for i, k in enumerate(n_levels):
        f = helmert(k) if i in subset else np.ones((1, k)) / np.sqrt(k)
        c = np.kron(c, f)
    return c


def rm_effect(y: np.ndarray, c: np.ndarray) -> dict:
    """F test of one within-subject effect with exact GG epsilon.

    y: subjects x cells; c: orthonormal contrast matrix (m x cells) of the effect.
    """
    n, m = y.shape[0], c.shape[0]
    z = y @ c.T
    s = np.atleast_2d(np.cov(z, rowvar=False))
    tr_s = float(np.trace(s))
    ss_eff = float(n * (z.mean(0) ** 2).sum())
    ss_err = float((n - 1) * tr_s)
    f = ss_eff / tr_s if tr_s > 0 else np.nan
    eps = tr_s**2 / (m * float(np.trace(s @ s)))
    d1, d2 = m, m * (n - 1)
    return {
        "SS": ss_eff, "ddof1": d1, "ddof2": d2, "MS": ss_eff / d1, "F": f,
        "p_unc": float(stats.f.sf(f, d1, d2)),
        "p_GG_corr": float(stats.f.sf(f, d1 * eps, d2 * eps)),
        "np2": f * d1 / (f * d1 + d2), "eps": float(eps), "SS_err": ss_err,
    }


def rm_anova_exact(data: pd.DataFrame, dv: str, within: str | list[str],
                   subject: str = "participant") -> pd.DataFrame:
    """One- or multi-way repeated-measures ANOVA (complete cases only), exact GG epsilon per effect.

    Returns one row per effect with pingouin-style columns:
    Source, SS, ddof1, ddof2, MS, F, p_unc, p_GG_corr, np2, eps, n_participants.
    ``p_GG_corr`` is *always* the GG-corrected p (equal to ``p_unc`` when eps = 1, i.e. 2-level effects).
    """
    within = [within] if isinstance(within, str) else list(within)
    d = data[[subject, dv, *within]].dropna()
    wide = d.pivot_table(index=subject, columns=within, values=dv, aggfunc="mean").sort_index(axis=1)
    n_levels = [int(d[w].nunique()) for w in within]
    if wide.shape[1] != int(np.prod(n_levels)):
        raise ValueError("Design has empty cells; cannot build the full within-subject table.")
    wide = wide.dropna()
    if len(wide) < 3:
        raise ValueError("Need at least 3 complete subjects.")
    y = wide.to_numpy()
    rows = []
    for r in range(1, len(within) + 1):
        for subset in combinations(range(len(within)), r):
            res = rm_effect(y, effect_contrasts(n_levels, subset))
            res.pop("SS_err")
            rows.append({"Source": " * ".join(within[i] for i in subset), **res,
                         "n_participants": int(len(wide))})
    return pd.DataFrame(rows)


__all__ = ["helmert", "effect_contrasts", "rm_effect", "rm_anova_exact"]
