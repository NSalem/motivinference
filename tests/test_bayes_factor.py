"""Tests of the default (JZS) Bayes factors the paper reports, all computed by pingouin.

``pingouin.bayesfactor_ttest`` computes Rouder et al. (2009), Eq. 1 with the Cauchy scale r of their Note 4
(r = 0.707 by default). These tests pin it, at the installed pingouin version, against:

1. R's ``BayesFactor`` (0.9.12-4.8, R 4.5.0): ``ttestBF`` on seeded raw data, and ``meta.ttestBF`` at (t, n) pairs
   behind BFs the paper reports and at two-sample (t, n1, n2) values. On 2026-11-19, pingouin matched
   ``meta.ttestBF`` within 4e-10 relative error at all 710 one-sample (t, n) pairs behind a reported BF.
2. The Bayes factors printed in Rouder et al. (2009) and Morey & Wagenmakers (2014, Table 1).
3. The convention of :func:`motinf.exploratory.paper_tables.paired_effect`: two-sided tests carry ``bf10``,
   one-tailed tests none (pingouin computes two-sided BFs only, and the paper reports no one-tailed BF).

A pingouin upgrade that changed these numbers fails here.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import pingouin as pg


from motinf.exploratory.paper_tables import paired_effect


def log_bf(t: float, n: int, n2: int | None = None, r: float = 0.707) -> float:
    return float(np.log(pg.bayesfactor_ttest(t, nx=n, ny=n2, r=r)))


class TestAgainstRRawData(unittest.TestCase):
    """``set.seed(1); x <- rnorm(30, 0.3, 1); ttestBF(x = x)`` returns 0.4080506 (default rscale
    "medium" = sqrt(2)/2). ``T`` is that sample's t statistic, computed with numpy from the same data.
    The 0.707 vs sqrt(2)/2 rounding moves BF by < 2e-4, inside the tolerance."""

    T, N = 1.2810925399388886, 30

    def test_two_sided(self) -> None:
        self.assertAlmostEqual(float(pg.bayesfactor_ttest(self.T, nx=self.N)), 0.4080506, delta=0.001)


class TestAgainstRSummaryStats(unittest.TestCase):
    """``extractBF(meta.ttestBF(t = t, n1 = n, [n2 = n2,] rscale = 0.707), logbf = TRUE)$bf``."""

    # (t, n, log BF10): (t, n) pairs behind BFs in paper_tables.json
    ONE_SAMPLE = [
        (0.0269360779, 216, -2.575404731758),
        (-1.9365016658, 216, -0.741122663288),
        (-2.6553397394, 84, 1.164418783795),
        (-6.0994, 84, 12.835174432917),
        (-1.5635514538, 132, -1.148996200649),
        (5.4934211916, 132, 10.986686935471),
    ]
    # (t, n1, n2, log BF10): two-sample (pooled-variance) t, the Experiment 1 vs 2 comparisons (n1 = 84, n2 = 132)
    TWO_SAMPLE = [
        (2.0, 84, 132, -0.019102303956),
        (-1.0, 84, 132, -1.415539946519),
        (3.5, 84, 132, 3.750161495148),
        (0.3, 30, 45, -1.375723023945),
    ]

    def test_one_sample(self) -> None:
        for t, n, ref in self.ONE_SAMPLE:
            with self.subTest(t=t, n=n):
                self.assertAlmostEqual(log_bf(t, n), ref, delta=1e-6)

    def test_two_sample(self) -> None:
        for t, n1, n2, ref in self.TWO_SAMPLE:
            with self.subTest(t=t, n1=n1, n2=n2):
                self.assertAlmostEqual(log_bf(t, n1, n2), ref, delta=1e-6)

    def test_large_t_merged_sample(self) -> None:
        # The largest |t| behind a reported BF (fast - slow, n = 216): finite and matching R.
        self.assertAlmostEqual(log_bf(-18.0490862458, 216), 95.6634117085205, delta=1e-6)


class TestAgainstPublishedValues(unittest.TestCase):
    def test_morey_wagenmakers_2014_table1(self) -> None:
        # B_e0^JZS column, n = 50, r = sqrt(2)/2 (computed in the paper with BayesFactor).
        for t, ref in ((0.11, 0.1547), (-1.57, 0.4835), (-0.29, 0.1601)):
            with self.subTest(t=t):
                self.assertAlmostEqual(float(pg.bayesfactor_ttest(t, nx=50, r=np.sqrt(2) / 2)), ref, delta=6e-5)

    def test_rouder_2009_examples(self) -> None:
        # Worked examples on pp. 232-233, reported as B01 = 1/BF10 with two decimals; r = 1 unless stated.
        for t, n, r, b01 in ((2.24, 80, 1.0, 1.02), (2.03, 80, 1.0, 1.56), (2.39, 48, 1.0, 0.63), (2.03, 80, 0.1, 0.59)):
            with self.subTest(t=t, n=n, r=r):
                self.assertAlmostEqual(1 / float(pg.bayesfactor_ttest(t, nx=n, r=r)), b01, delta=0.006)

    def test_rouder_2009_table1(self) -> None:
        # Critical t values (two decimals) for B01 = 3 and 1/10 at r = 1, Table 1 p. 232.
        for n, t3, t_tenth in ((10, 0.89, 3.60), (50, 1.51, 3.17), (100, 1.72, 3.20), (500, 2.12, 3.38)):
            with self.subTest(n=n):
                self.assertAlmostEqual(1 / float(pg.bayesfactor_ttest(t3, nx=n, r=1.0)), 3.0, delta=0.03)
                self.assertAlmostEqual(1 / float(pg.bayesfactor_ttest(t_tenth, nx=n, r=1.0)), 0.1, delta=0.002)


class TestPairedEffectConvention(unittest.TestCase):
    DIFF = pd.Series(np.random.default_rng(0).normal(0.3, 1.0, 40))

    def test_two_sided_has_bf(self) -> None:
        res = paired_effect(self.DIFF)
        self.assertAlmostEqual(res["bf10"], float(pg.bayesfactor_ttest(res["t"], nx=res["n"])), delta=0)

    def test_one_tailed_has_no_bf(self) -> None:
        for tail in ("greater", "less"):
            with self.subTest(tail=tail):
                self.assertNotIn("bf10", paired_effect(self.DIFF, tail=tail))


if __name__ == "__main__":
    unittest.main()
