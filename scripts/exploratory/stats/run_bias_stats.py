"""Common response biases (Supplementary Figure 7, Supplementary Table 6).

- Estimation, cardinal and serial bias: kernel-smoothed bias against stimulus angle (cardinal, 0-180 deg) and
  against the current - previous stimulus angle (serial), per participant x incentive; mean |bias| across
  the angle grid (degrees).
- Integration, temporal weighting: ``fit_sequential_bias`` (L2 logistic regression of choice on the
  lag-aligned sample evidence), one weight per participant x incentive x sample position.
- Integration, choice history: probit ``choice ~ sumllr_noisy + prev_stim + prev_choice + prev_resp_side``
  (lags within session) per participant x incentive.

The incentive tests (Supplementary Table 6, and the stars of Supplementary Figure 7) are computed from these tables
by ``paper_tables.bias_measures``.

Inputs: ``data/processed/<exp>/data_clean_{est,inf}.csv`` (pooled for ``--exp-name merged``).
Outputs in ``results/exploratory/<exp>/modeling/stats/bias_stats/params/``: ``estimation_mean_abs_dev.csv``,
``temporal_beta_long.csv`` and ``probit_carryover_coefs_long.csv``, plus, for the pooled sample, the bias curves drawn
in Supplementary Figure 7a (``estimation_ksr_curves_long.csv``). Figure: ``scripts/exploratory/figures/sfig7.py``.

Usage::

    python scripts/exploratory/stats/run_bias_stats.py --exp-name merged
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit

ROOT = Path(__file__).resolve().parents[3]

from motinf.exploratory.common_biases import (
    add_estimation_serial_columns,
    aggregate_mean_abs_dev_from_long,
    bias_per_ori_to_long_df,
    cardinal_estimation_mean_angles,
    fit_sequential_bias,
    get_ksr_curves,
    prep_inf_for_categ_biases,
)
from motinf.exploratory.cohorts import cohort_subexperiments, merge_data_clean_est, merge_data_clean_inf


PROBIT_COEF_COLS = ("b_0", "b_stim", "b_prev_stim", "b_prev_resp", "b_prev_key")
SEQUENTIAL_BIAS_C = 0.1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--exp-name",
        type=str,
        default="merged",
        help="exp1, exp2 or merged.",
    )
    return p.parse_args()


def _run_estimation_ksr_cardinal_serial(df_est: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Kernel-smoothed cardinal and serial bias curves (long table) and their mean |bias| per participant x incentive."""
    data_mean = cardinal_estimation_mean_angles(df_est)
    x_card = np.linspace(0.0, 180.0, 181)
    res_card = get_ksr_curves(
        data_mean,
        x_var_name="estimang_deg",
        y_var_name="dev_deg",
        x_vals_out=x_card,
        bw=[10],
        verbose=False,
    )
    inc = np.sort(data_mean["incentive"].unique())
    pids = data_mean["participant"].unique()
    long_card = bias_per_ori_to_long_df(
        res_card["bias_per_ori"], inc, pids, x_card, task="cardinal"
    )
    mad_card = aggregate_mean_abs_dev_from_long(long_card)

    df_serial = add_estimation_serial_columns(df_est.copy())
    df_serial["rel_prev_ang_deg"] = np.rad2deg(df_serial["rel_prev_ang"]).round(1)
    x_ser = np.linspace(-90.0, 90.0, 180)
    res_ser = get_ksr_curves(
        df_serial,
        x_var_name="rel_prev_ang_deg",
        y_var_name="dev_demean_deg",
        x_vals_out=x_ser,
        bw=[10],
        verbose=False,
    )
    long_ser = bias_per_ori_to_long_df(
        res_ser["bias_per_ori"], inc, pids, x_ser, task="serial"
    )
    mad_ser = aggregate_mean_abs_dev_from_long(long_ser)

    curves_long = pd.concat([long_card, long_ser], ignore_index=True)
    mean_abs = pd.concat([mad_card, mad_ser], ignore_index=True)
    return curves_long, mean_abs


def _fit_probit_carryover_long(df_inf: pd.DataFrame) -> pd.DataFrame:
    """Per participant × incentive probit coefficients (lags computed within session; columns of the design in the
    order of ``PROBIT_COEF_COLS``). A cell without a fit gets NaN coefficients and the reason in ``error``."""
    rows: list[dict] = []

    def add(pid: str, iinc, values, error: str | None = None) -> None:
        """One row per coefficient; ``error`` None for a fit, else the reason there is none."""
        rows.extend({"participant": pid, "incentive": float(iinc), "coef": cname, "value": value,
                     "fit_ok": error is None, "error": "" if error is None else error}
                    for cname, value in zip(PROBIT_COEF_COLS, values))

    nan_values = [np.nan] * len(PROBIT_COEF_COLS)
    participants = sorted(df_inf["participant"].astype(str).unique())
    incentives = np.sort(df_inf["incentive"].unique())
    for pid in participants:
        for iinc in incentives:
            df_sub = df_inf[
                (df_inf["participant"].astype(str) == pid) & (df_inf["incentive"] == iinc)
            ].sort_values(["session", "trialN"], kind="mergesort").reset_index(drop=True)
            df_sub = df_sub.copy()
            df_sub["_c01"] = df_sub["choice"].map({-1: 0, 1: 1})
            g2 = df_sub.groupby("session", sort=False)
            df_sub["prev_stim"] = g2["sumllr_noisy"].shift(1)
            df_sub["prev_choice"] = g2["_c01"].shift(1)
            df_sub.drop(columns=["_c01"], inplace=True)
            msk = df_sub[["prev_stim", "prev_choice", "prev_resp_side"]].notna().all(axis=1)
            if msk.sum() < 10:
                add(pid, iinc, nan_values, "too_few_trials")
                continue
            X = df_sub.loc[msk, ["sumllr_noisy", "prev_stim", "prev_choice", "prev_resp_side"]].to_numpy(
                dtype=float
            )
            X = sm.add_constant(X, has_constant="add")
            y = df_sub.loc[msk, "choice"].map({-1: 0, 1: 1}).to_numpy(dtype=float)
            if np.unique(y).size < 2:
                add(pid, iinc, nan_values, "single_class_y")
                continue
            try:
                results = Probit(y, X).fit(disp=0)
                add(pid, iinc, [float(results.params[j]) for j in range(len(PROBIT_COEF_COLS))])
            except Exception as exc:  # pragma: no cover
                add(pid, iinc, nan_values, str(exc)[:200])
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    cohort_segs = cohort_subexperiments(args.exp_name)
    params_dir = ROOT / "results" / "exploratory" / args.exp_name / "modeling" / "stats" / "bias_stats" / "params"
    params_dir.mkdir(parents=True, exist_ok=True)

    df_est = merge_data_clean_est(ROOT, cohort_segs)
    df_inf = prep_inf_for_categ_biases(merge_data_clean_inf(ROOT, cohort_segs))

    curves_long, mad = _run_estimation_ksr_cardinal_serial(df_est)
    if len(cohort_segs) > 1:  # the curves are drawn for the pooled sample only (Supp. Fig. 7a)
        curves_long.to_csv(params_dir / "estimation_ksr_curves_long.csv", index=False)
    mad.to_csv(params_dir / "estimation_mean_abs_dev.csv", index=False)
    fit_sequential_bias(df_inf, C=SEQUENTIAL_BIAS_C).to_csv(params_dir / "temporal_beta_long.csv", index=False)
    _fit_probit_carryover_long(df_inf).to_csv(params_dir / "probit_carryover_coefs_long.csv", index=False)
    print(f"Wrote {params_dir}")


if __name__ == "__main__":
    main()
