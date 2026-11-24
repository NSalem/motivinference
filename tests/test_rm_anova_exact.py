"""Tests for motinf.prereg.rm_anova_exact (exact Greenhouse-Geisser epsilon).

What is pinned:

1. One-way designs and main effects of two-way designs reproduce pingouin exactly
   (F, epsilon, uncorrected and GG-corrected p).
2. The two-way *interaction* F and dfs match pingouin, but epsilon deliberately does not
   (pingouin's interaction epsilon is inaccurate when both factors have > 2 levels).
3. The exact interaction epsilon equals an independent construction (covariance of the
   double-centred cell table, which lives in the interaction subspace) and is invariant to the
   choice of orthonormal contrast basis.
4. Two-level effects have epsilon == 1 (GG p == uncorrected p).
"""

from __future__ import annotations

import unittest
import warnings

import numpy as np
import pandas as pd


from motinf.prereg.rm_anova_exact import (
    effect_contrasts,
    rm_anova_exact,
    rm_effect,
)

try:
    import pingouin as pg
except ImportError:  # pragma: no cover
    pg = None


def _long(y: np.ndarray, names: list[str], levels: list[list[float]]) -> pd.DataFrame:
    """y: subjects x cells (factor-major order) -> long DataFrame."""
    n = y.shape[0]
    cells = [(a, b) for a in levels[0] for b in levels[1]] if len(levels) == 2 else [(a,) for a in levels[0]]
    rows = []
    for s in range(n):
        for j, cell in enumerate(cells):
            rows.append({"participant": f"s{s}", "dv": y[s, j], **dict(zip(names, cell))})
    return pd.DataFrame(rows)


def _simulate(n=60, k=9, seed=1):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(k, k))
    cov = a @ a.T / k + np.diag(np.linspace(0.2, 2.0, k))  # non-spherical
    mu = np.linspace(0, 0.8, k)
    return rng.multivariate_normal(mu, cov, size=n)


class RmAnovaExactTests(unittest.TestCase):
    def _pg(self, df, within):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = pg.rm_anova(data=df, dv="dv", within=within, subject="participant",
                            detailed=True, correction=True, effsize="np2")
        return r.rename(columns=lambda c: c.replace("-", "_"))

    @unittest.skipIf(pg is None, "pingouin not installed")
    def test_one_way_matches_pingouin(self):
        y = _simulate(k=4, seed=2)
        df = _long(y, ["cond"], [[1.0, 2.0, 3.0, 4.0]])
        ours = rm_anova_exact(df, "dv", "cond").iloc[0]
        ref = self._pg(df, "cond").iloc[0]
        self.assertAlmostEqual(ours["F"], ref["F"], places=8)
        self.assertAlmostEqual(ours["eps"], ref["eps"], places=8)
        self.assertAlmostEqual(ours["p_unc"], ref["p_unc"], places=10)
        self.assertAlmostEqual(ours["p_GG_corr"], ref["p_GG_corr"], places=10)
        self.assertAlmostEqual(ours["np2"], ref["np2"], places=8)

    @unittest.skipIf(pg is None, "pingouin not installed")
    def test_two_way_main_effects_match_pingouin_interaction_f_matches_but_eps_differs(self):
        y = _simulate(k=9, seed=3)
        df = _long(y, ["a", "b"], [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        ours = rm_anova_exact(df, "dv", ["a", "b"]).set_index("Source")
        ref = self._pg(df, ["a", "b"]).set_index("Source")
        for src in ("a", "b"):
            self.assertAlmostEqual(ours.loc[src, "F"], ref.loc[src, "F"], places=8)
            self.assertAlmostEqual(ours.loc[src, "eps"], ref.loc[src, "eps"], places=8)
            self.assertAlmostEqual(ours.loc[src, "p_GG_corr"], ref.loc[src, "p_GG_corr"], places=10)
        self.assertAlmostEqual(ours.loc["a * b", "F"], ref.loc["a * b", "F"], places=8)
        self.assertEqual(int(ours.loc["a * b", "ddof1"]), int(ref.loc["a * b", "ddof1"]))
        self.assertEqual(int(ours.loc["a * b", "ddof2"]), int(ref.loc["a * b", "ddof2"]))
        # the known pingouin limitation: interaction epsilon differs (that is the point of this module)
        self.assertGreater(abs(ours.loc["a * b", "eps"] - ref.loc["a * b", "eps"]), 0.05)

    def test_interaction_eps_equals_independent_double_centred_construction(self):
        y = _simulate(k=9, seed=4)
        n = y.shape[0]
        cube = y.reshape(n, 3, 3)
        dc = (cube - cube.mean(2, keepdims=True) - cube.mean(1, keepdims=True)
              + cube.mean((1, 2), keepdims=True)).reshape(n, 9)
        s9 = np.cov(dc, rowvar=False)
        eps_alt = np.trace(s9) ** 2 / (4 * np.trace(s9 @ s9))
        got = rm_effect(y, effect_contrasts([3, 3], (0, 1)))
        self.assertAlmostEqual(got["eps"], eps_alt, places=10)

    def test_eps_and_f_invariant_to_orthonormal_basis(self):
        y = _simulate(k=9, seed=5)
        c = effect_contrasts([3, 3], (0, 1))
        rng = np.random.default_rng(0)
        q, _ = np.linalg.qr(rng.normal(size=(c.shape[0], c.shape[0])))
        a, b = rm_effect(y, c), rm_effect(y, q @ c)
        self.assertAlmostEqual(a["eps"], b["eps"], places=10)
        self.assertAlmostEqual(a["F"], b["F"], places=10)

    def test_two_level_effect_has_unit_epsilon(self):
        y = _simulate(k=2, seed=6)
        df = _long(y, ["cond"], [[0.0, 1.0]])
        r = rm_anova_exact(df, "dv", "cond").iloc[0]
        self.assertAlmostEqual(r["eps"], 1.0, places=12)
        self.assertAlmostEqual(r["p_GG_corr"], r["p_unc"], places=12)

    def test_incomplete_subjects_are_dropped(self):
        y = _simulate(k=3, seed=7)
        df = _long(y, ["cond"], [[1.0, 2.0, 3.0]])
        df = df.drop(df.index[0])  # subject s0 loses one cell
        r = rm_anova_exact(df, "dv", "cond").iloc[0]
        self.assertEqual(int(r["n_participants"]), y.shape[0] - 1)


if __name__ == "__main__":
    unittest.main()
