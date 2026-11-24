"""Unit tests: R² of observed vs predicted choice variability (Results), correlations of the full-model δ."""

from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd


from motinf.exploratory.behavior_stats import r2_obs_vs_pred_choice_var
from motinf.exploratory.full_model_results import compute_motiv_correlation_table


class TestR2ObsVsPred(unittest.TestCase):
    def test_perfect_fit(self) -> None:
        y = np.array([1.0, 4.0, 9.0])
        r2, n = r2_obs_vs_pred_choice_var(y, y.copy())
        self.assertEqual(n, 3)
        self.assertAlmostEqual(r2, 1.0, places=10)

    def test_intercept_only_benchmark(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        pred = np.full(3, np.mean(y))
        r2, n = r2_obs_vs_pred_choice_var(y, pred)
        self.assertEqual(n, 3)
        self.assertAlmostEqual(r2, 0.0, places=10)

    def test_constant_y_nan(self) -> None:
        y = np.array([3.0, 3.0, 3.0])
        pred = np.array([1.0, 2.0, 4.0])
        r2, n = r2_obs_vs_pred_choice_var(y, pred)
        self.assertEqual(n, 3)
        self.assertTrue(math.isnan(r2))

    def test_single_cell_nan(self) -> None:
        r2, n = r2_obs_vs_pred_choice_var(np.array([1.0]), np.array([1.0]))
        self.assertEqual(n, 1)
        self.assertTrue(math.isnan(r2))

    def test_masks_nonfinite(self) -> None:
        y = np.array([1.0, np.nan, 3.0])
        p = np.array([1.1, 2.0, 3.1])
        r2, n = r2_obs_vs_pred_choice_var(y, p)
        self.assertEqual(n, 2)
        self.assertTrue(np.isfinite(r2))


class TestMotivCorrelationTable(unittest.TestCase):
    def test_pairwise_motiv_correlations(self) -> None:
        df = pd.DataFrame(
            {
                "motiv_ang": [0.0, 1.0, 2.0, 3.0],
                "motiv_inf": [0.0, 1.0, 2.0, 3.0],
                "motiv_sel": [0.0, 2.0, 4.0, 6.0],
            }
        )
        out = compute_motiv_correlation_table(df)
        self.assertEqual(len(out), 3)
        self.assertTrue({(row["x"], row["y"]) for _, row in out.iterrows()}.issuperset(
            {("motiv_ang", "motiv_inf"), ("motiv_inf", "motiv_sel"), ("motiv_ang", "motiv_sel")}))
        pair = out[(out["x"] == "motiv_ang") & (out["y"] == "motiv_inf")].iloc[0]
        self.assertAlmostEqual(float(pair["pearson_r"]), 1.0, places=10)
        self.assertAlmostEqual(float(pair["spearman_r"]), 1.0, places=10)

    def test_missing_column_raises(self) -> None:
        df = pd.DataFrame({"motiv_ang": [0.0, 1.0], "motiv_inf": [0.0, 1.0]})
        with self.assertRaises(KeyError):
            compute_motiv_correlation_table(df)


if __name__ == "__main__":
    unittest.main()
