"""Statistics conventions of the paper, each against an independent computation.

1. The exact Greenhouse–Geisser interaction of a 3 × 3 within-subject design (sequence length × incentive, H2):
   F, its degrees of freedom, ε and the corrected P from the textbook sums-of-squares decomposition
   (``motinf.prereg.rm_anova_exact``). The other ANOVA cases are in ``test_rm_anova_exact.py``.
2. dz with its exact confidence interval (``motinf.exploratory.stats.dz_ci``): the large-sample normal interval and
   the coverage of simulated samples.
3. The Holm correction (``motinf.exploratory.stats.holm``) against statsmodels.

Bayes factors, the variance partition and the recovery of the correlation ρ between repetitions are tested in
``test_bayes_factor.py`` and ``test_variance_partition_*.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from motinf.exploratory.stats import dz_ci, holm
from motinf.prereg.rm_anova_exact import rm_anova_exact


def _two_way(n: int, seed: int) -> np.ndarray:
    """n participants × 3 × 3 cells, non-spherical covariance, an interaction."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(9, 9))
    cov = a @ a.T / 9 + np.diag(np.linspace(0.2, 2.0, 9))
    mu = (np.outer([0, 0.3, 0.6], [1, 1.2, 1.5]) + np.array([[0, 0, 0], [0, 0.2, 0], [0, 0, 0.4]])).ravel()
    return rng.multivariate_normal(mu, cov, size=n).reshape(n, 3, 3)


def _textbook_interaction(y: np.ndarray) -> dict:
    """A × B of an n × a × b within-subject design: sums of squares, ε from the double-centred covariance."""
    n, a, b = y.shape
    m, m_s = y.mean(), y.mean(axis=(1, 2))
    m_a, m_b, m_ab = y.mean(axis=(0, 2)), y.mean(axis=(0, 1)), y.mean(axis=0)
    m_sa, m_sb = y.mean(axis=2), y.mean(axis=1)
    ss_ab = n * np.sum((m_ab - m_a[:, None] - m_b[None, :] + m) ** 2)
    resid = (y - m_sa[:, :, None] - m_sb[:, None, :] - m_ab[None] + m_s[:, None, None] + m_a[None, :, None]
             + m_b[None, None, :] - m)
    df1, df2 = (a - 1) * (b - 1), (a - 1) * (b - 1) * (n - 1)
    f = (ss_ab / df1) / (np.sum(resid**2) / df2)
    dc = (y - y.mean(2, keepdims=True) - y.mean(1, keepdims=True) + y.mean((1, 2), keepdims=True)).reshape(n, a * b)
    s = np.cov(dc, rowvar=False)
    eps = np.trace(s) ** 2 / (df1 * np.trace(s @ s))
    return {"F": f, "ddof1": df1, "ddof2": df2, "eps": eps, "p_GG_corr": stats.f.sf(f, eps * df1, eps * df2)}


@pytest.mark.parametrize("n, seed", [(84, 1), (132, 2), (20, 3)])
def test_3x3_interaction_against_the_textbook_decomposition(n: int, seed: int) -> None:
    y = _two_way(n, seed)
    rows = [{"participant": s, "a": i, "b": j, "dv": y[s, i, j]} for s in range(n) for i in range(3) for j in range(3)]
    got = rm_anova_exact(pd.DataFrame(rows), "dv", ["a", "b"]).set_index("Source").loc["a * b"]
    ref = _textbook_interaction(y)
    assert int(got["ddof1"]) == ref["ddof1"] and int(got["ddof2"]) == ref["ddof2"]
    for k in ("F", "eps", "p_GG_corr"):
        assert got[k] == pytest.approx(ref[k], rel=1e-9), k


def test_dz_ci_matches_the_large_sample_normal_interval() -> None:
    n, d = 20000, 0.3
    dz, lo, hi = dz_ci(d * np.sqrt(n), n)
    half = stats.norm.ppf(0.975) * np.sqrt(1 / n + d**2 / (2 * n))  # SE of a standardized mean
    assert dz == pytest.approx(d)
    assert lo == pytest.approx(d - half, abs=2e-4) and hi == pytest.approx(d + half, abs=2e-4)


def test_dz_ci_is_symmetric_in_the_sign_of_t() -> None:
    dz, lo, hi = dz_ci(3.29, 84)
    assert dz_ci(-3.29, 84) == pytest.approx((-dz, -hi, -lo), rel=1e-9)
    assert lo < dz < hi


def test_dz_ci_has_its_nominal_coverage() -> None:
    rng = np.random.default_rng(0)
    n, d, reps = 25, 0.5, 600
    covered = 0
    for _ in range(reps):
        x = rng.normal(d, 1, n)
        t = x.mean() / (x.std(ddof=1) / np.sqrt(n))
        _, lo, hi = dz_ci(t, n)
        covered += lo <= d <= hi
    assert abs(covered / reps - 0.95) < 3.5 * np.sqrt(0.95 * 0.05 / reps)


def test_holm_matches_statsmodels() -> None:
    multipletests = pytest.importorskip("statsmodels.stats.multitest").multipletests
    rng = np.random.default_rng(1)
    for k in (2, 3, 6, 12):
        ps = rng.uniform(0, 0.2, k) ** 2
        np.testing.assert_allclose(holm(list(ps)), multipletests(ps, method="holm")[1], rtol=1e-12)


def test_holm_leaves_a_missing_p_missing_and_does_not_count_it() -> None:
    got = holm([0.01, float("nan"), 0.04])
    assert np.isnan(got[1])
    assert got[0] == pytest.approx(0.02) and got[2] == pytest.approx(0.04)
