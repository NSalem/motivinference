"""Tests for motinf.exploratory.variance_partition.fit_bias_share_repeats.

What is pinned:

1. The pair likelihood equals the bivariate-normal orthant probabilities (scipy's MVN CDF).
2. The Gauss-Hermite path for sequences shown 3+ times agrees with the exact pair path when
   given pairs, and with Monte Carlo for a triple.
3. Recovery on synthetic data: the MLE lands near the true rho and its 95% profile-likelihood
   interval covers it; the Beta(1.1, 1.1) MAP stays strictly inside (0, 1).
4. With independent presentations (rho = 0) the interval reaches 0 while the MAP stays positive.
5. Sequences shown once do not change the fit; inconsistent evidence within a sequence raises.
"""

from __future__ import annotations

import unittest

import numpy as np
from scipy import stats


from motinf.exploratory.variance_partition import _repeat_loglik, fit_bias_share_repeats


def _simulate(rho: float, n_seq: int, n_pres: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    h = rng.normal(0, 1.2, n_seq)
    shared = rng.normal(size=n_seq)
    hs, choices, groups = [], [], []
    for _ in range(n_pres):
        e = np.sqrt(rho) * shared + np.sqrt(1 - rho) * rng.normal(size=n_seq)
        hs.append(h); choices.append(h + e > 0); groups.append(np.arange(n_seq))
    return np.concatenate(hs), np.concatenate(choices), np.concatenate(groups)


class TestFitBiasShareRepeats(unittest.TestCase):
    def test_pair_probabilities_match_bivariate_normal(self) -> None:
        h = np.array([-1.3, -0.2, 0.0, 0.7, 2.1])
        for rho in (0.0, 0.3, 0.75):
            mvn = stats.multivariate_normal([0, 0], [[1, rho], [rho, 1]])
            for n_a in (0, 1, 2):
                exp = []
                for hi in h:
                    paa = mvn.cdf([hi, hi])                      # both decision variables > 0 given mean hi
                    pa = stats.norm.cdf(hi)
                    exp.append({2: paa, 1: pa - paa, 0: 1 - 2 * pa + paa}[n_a])
                got = _repeat_loglik(rho, (h, np.full(h.size, n_a)), (np.empty(0),) * 3)[0]
                np.testing.assert_allclose(got, np.sum(np.log(exp)), rtol=1e-6)

    def test_quadrature_path(self) -> None:
        h = np.array([-0.8, 0.1, 1.5]); n_a = np.array([0, 1, 2])
        for rho in (0.0, 0.3, 0.8):
            exact = _repeat_loglik(rho, (h, n_a), (np.empty(0),) * 3)[0]
            quad = _repeat_loglik(rho, (np.empty(0), np.empty(0)), (h, n_a.astype(float), np.full(3, 2.0)))[0]
            np.testing.assert_allclose(quad, exact, rtol=1e-8)
        # a triple with two 'A' choices, against Monte Carlo
        rng = np.random.default_rng(0)
        rho, hi = 0.4, 0.5
        e = np.sqrt(rho) * rng.normal(size=(1, 400_000)) + np.sqrt(1 - rho) * rng.normal(size=(3, 400_000))
        c = (hi + e) > 0
        p_mc = np.mean(c.sum(axis=0) == 2) / 3                    # one specific ordering
        quad = np.exp(_repeat_loglik(rho, (np.empty(0), np.empty(0)), (np.array([hi]), np.array([2.0]), np.array([3.0])))[0])
        np.testing.assert_allclose(quad, p_mc, rtol=0.02)

    def test_recovery(self) -> None:
        h, c, g = _simulate(0.4, 150, 2, seed=1)
        h3, c3, g3 = _simulate(0.4, 10, 3, seed=2)
        res = fit_bias_share_repeats(np.r_[h, h3], np.r_[c, c3], np.r_[g, g3 + 1000])
        self.assertEqual((res["n_pairs"], res["n_multi"]), (150, 10))
        self.assertLess(abs(res["rho_mle"] - 0.4), 0.2)
        self.assertLessEqual(res["ci_lo"], 0.4); self.assertGreaterEqual(res["ci_hi"], 0.4)
        self.assertTrue(0 < res["rho_map"] < 1)
        self.assertEqual(res["loglik"].shape, (200,))

    def test_independent_presentations(self) -> None:
        h, c, g = _simulate(0.0, 150, 2, seed=3)
        res = fit_bias_share_repeats(h, c, g)
        self.assertEqual(res["ci_lo"], 0.0)
        self.assertGreater(res["rho_map"], 0.0)
        self.assertLess(res["rho_mle"], 0.25)

    def test_singletons_and_consistency(self) -> None:
        h, c, g = _simulate(0.3, 80, 2, seed=4)
        base = fit_bias_share_repeats(h, c, g)
        with_single = fit_bias_share_repeats(np.r_[h, 0.3, -1.0], np.r_[c, True, False], np.r_[g, 5000, 5001])
        self.assertAlmostEqual(base["rho_map"], with_single["rho_map"], places=10)
        bad_h = h.copy(); bad_h[0] += 0.5                        # same sequence, different evidence
        with self.assertRaises(ValueError):
            fit_bias_share_repeats(bad_h, c, g)


class TestSplitTotalVariance(unittest.TestCase):
    def test_clipping_and_floor(self) -> None:
        from motinf.exploratory.variance_partition import split_total_variance

        rho = np.array([-0.2, 0.0, 0.4, 1.0, np.nan])
        tv = np.array([4.0, 4.0, 4.0, 4.0, 4.0])
        s = split_total_variance(rho, tv)
        np.testing.assert_allclose(s["rho"][:4], [1e-3, 1e-3, 0.4, 0.999])
        np.testing.assert_allclose(s["bias_var"][:4] + s["noise_var"][:4], 4.0)
        np.testing.assert_allclose(s["bias_sd"][2], np.sqrt(1.6))
        self.assertTrue(np.all(np.isnan([s[k][4] for k in ("rho", "bias_sd", "noise_sd", "bias_var")])))
        tiny = split_total_variance(0.5, 1e-12)                  # SDs floored at 1e-4
        self.assertEqual((float(tiny["bias_sd"]), float(tiny["noise_sd"]), float(tiny["total_sd"])), (1e-4, 1e-4, 1e-4))
        np.testing.assert_allclose(tiny["bias_var"], 1e-8)


if __name__ == "__main__":
    unittest.main()
