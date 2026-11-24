"""Tests for motinf.exploratory.variance_partition.log_decomposition_partition.

What is pinned:

1. Bars equal a direct per-participant computation of the log decomposition, including a small
   worked example (two participants, one reducing only noise, the other only bias).
2. The reference bar is the mean composition and sums to 1; every other bar sums to
   exp(mean log total ratio).
3. With a single participant the bars are exactly component / reference total.
4. A component that does not change in any participant keeps its reference segment exactly.
5. Each participant counts equally: rescaling one participant's variances changes nothing, and
   equal-and-opposite changes (x2, x0.5) cancel.
6. Components equal to 0 are fine (only totals are logged); invalid participants are dropped.
7. Bootstrap replicates have the right shape, are reproducible for a seed and keep the reference
   bar at 1.
"""

from __future__ import annotations

import unittest

import numpy as np


from motinf.exploratory.variance_partition import log_decomposition_partition


def _random_partition(n: int = 40, k: int = 3, c: int = 3, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.gamma(2.0, 1.0, size=(n, k, c)) * rng.lognormal(0.0, 1.0, size=(n, 1, 1))


def _logmean(a: float, b: float) -> float:
    return a if np.isclose(a, b, rtol=1e-14, atol=0) else (a - b) / (np.log(a) - np.log(b))


def _direct(comp: np.ndarray, ref: int) -> np.ndarray:
    """Loop implementation straight from the definition."""
    n, k, c = comp.shape
    total = comp.sum(axis=1)
    share = np.mean([comp[i, :, ref] / total[i, ref] for i in range(n)], axis=0)
    bar = np.empty((k, c))
    for j in range(c):
        d = np.mean(
            [(comp[i, :, j] - comp[i, :, ref]) / _logmean(total[i, j], total[i, ref]) for i in range(n)],
            axis=0,
        )
        g = np.exp(np.mean(np.log(total[:, j] / total[:, ref])))
        bar[:, j] = share + _logmean(g, 1.0) * d
    return bar


class TestLogDecompositionPartition(unittest.TestCase):
    def test_matches_direct_computation(self) -> None:
        comp = _random_partition()
        bar, _, keep = log_decomposition_partition(comp, ref=1, n_boot=0)
        self.assertTrue(keep.all())
        np.testing.assert_allclose(bar, _direct(comp, ref=1))

    def test_worked_example(self) -> None:
        # A: noise 0.8 -> 0.6, bias 0.2 -> 0.2; B: noise 2 -> 2, bias 2 -> 1
        comp = np.array([[[0.8, 0.6], [0.2, 0.2]], [[2.0, 2.0], [2.0, 1.0]]])
        bar, _, _ = log_decomposition_partition(comp, ref=0, n_boot=0)
        np.testing.assert_allclose(bar[:, 0], [0.65, 0.35])
        # A contributes ln(0.8) to noise only, B ln(0.75) to bias only; G = sqrt(0.8 * 0.75)
        g = np.sqrt(0.8 * 0.75)
        k = (g - 1) / np.log(g)
        np.testing.assert_allclose(bar[:, 1], [0.65 + k * np.log(0.8) / 2, 0.35 + k * np.log(0.75) / 2])
        np.testing.assert_allclose(bar[:, 1], [0.5515, 0.2231], atol=1e-4)
        np.testing.assert_allclose(bar[:, 1].sum(), g)

    def test_reference_sums_to_one_and_others_to_geometric_ratio(self) -> None:
        comp = _random_partition()
        bar, _, _ = log_decomposition_partition(comp, ref=0, n_boot=0)
        total = comp.sum(axis=1)
        np.testing.assert_allclose(bar[:, 0], np.mean(comp[:, :, 0] / total[:, [0]], axis=0))
        np.testing.assert_allclose(bar.sum(axis=0), np.exp(np.mean(np.log(total / total[:, [0]]), axis=0)))

    def test_single_participant_is_exact(self) -> None:
        comp = _random_partition(n=1)
        bar, _, _ = log_decomposition_partition(comp, ref=2, n_boot=0)
        np.testing.assert_allclose(bar, comp[0] / comp[0, :, 2].sum())

    def test_unchanged_component_stays_flat(self) -> None:
        comp = _random_partition()
        comp[:, 2, :] = comp[:, 2, [0]]  # component 2 identical in every condition
        bar, _, _ = log_decomposition_partition(comp, ref=0, n_boot=0)
        np.testing.assert_allclose(bar[2], bar[2, 0])
        self.assertFalse(np.allclose(bar[0], bar[0, 0]))

    def test_participants_weighted_equally(self) -> None:
        comp = _random_partition()
        scaled = comp.copy()
        scaled[0] *= 1e4
        np.testing.assert_allclose(
            log_decomposition_partition(scaled, ref=0, n_boot=0)[0],
            log_decomposition_partition(comp, ref=0, n_boot=0)[0],
        )
        comp2 = np.array([[[1.0, 2.0], [3.0, 6.0]], [[1.0, 0.5], [3.0, 1.5]]])
        bar, _, _ = log_decomposition_partition(comp2, ref=0, n_boot=0)
        np.testing.assert_allclose(bar, np.array([[0.25, 0.25], [0.75, 0.75]]))

    def test_zero_components_and_invalid_participants(self) -> None:
        comp = _random_partition(n=6)
        comp[2, 1, 0] = 0.0  # zero component: fine
        comp[4, 1, :] = 0.0
        bar, _, keep = log_decomposition_partition(comp, ref=0, n_boot=0)
        self.assertTrue(keep.all() and np.all(np.isfinite(bar)))
        np.testing.assert_allclose(bar, _direct(comp, ref=0))
        comp[1] = 0.0  # zero total: dropped
        comp[3, 0, 2] = np.nan  # non-finite: dropped
        bar, _, keep = log_decomposition_partition(comp, ref=0, n_boot=0)
        np.testing.assert_array_equal(keep, [True, False, True, False, True, True])
        np.testing.assert_allclose(bar, log_decomposition_partition(comp[keep], ref=0, n_boot=0)[0])

    def test_bootstrap(self) -> None:
        comp = _random_partition(n=60)
        bar, boot, _ = log_decomposition_partition(comp, ref=0, n_boot=2000, seed=3)
        self.assertEqual(boot.shape, (2000, *bar.shape))
        np.testing.assert_array_equal(boot, log_decomposition_partition(comp, ref=0, n_boot=2000, seed=3)[1])
        np.testing.assert_allclose(boot[:, :, 0].sum(axis=1), 1.0)
        np.testing.assert_allclose(boot.mean(axis=0), bar, atol=0.02)


if __name__ == "__main__":
    unittest.main()
