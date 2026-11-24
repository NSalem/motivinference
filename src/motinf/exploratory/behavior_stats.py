"""Behaviour statistics outside the pre-registered pipeline, and observed-vs-predicted choice
variability.

Entry point :func:`run_behavior_stats` (``scripts/exploratory/stats/run_behavior_stats.py``);
:func:`var_red_by_seqlen_r2_per_participant` is also used for the model comparisons of
Figure 3d and Supplementary Figure 4c.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm, ttest_1samp

from motinf.exploratory.behav_stats_paths import behav_extra_dir
from motinf.exploratory.cohorts import (
    cohort_subexperiments,
    merge_inf_trials_for_choice_var,
    merge_sd_inf_long,
    validate_prereg_inputs_exist,
)
from motinf.exploratory.choice_var_predictions import predict_inf_choicevar_per_seqlen
from motinf.prereg.rm_anova_exact import rm_anova_exact


def r2_obs_vs_pred_choice_var(
    y_obs: np.ndarray, y_pred: np.ndarray
) -> tuple[float, int]:
    """1 - SS_res/SS_tot for vectors of equal length; NaN if SS_tot==0 or n<2."""
    y_obs = np.asarray(y_obs, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_obs) & np.isfinite(y_pred)
    y_obs = y_obs[ok]
    y_pred = y_pred[ok]
    n = int(y_obs.size)
    if n < 2:
        return float("nan"), n
    ym = float(np.mean(y_obs))
    ss_tot = float(np.sum((y_obs - ym) ** 2))
    if ss_tot <= 0.0:
        return float("nan"), n
    ss_res = float(np.sum((y_obs - y_pred) ** 2))
    return 1.0 - ss_res / ss_tot, n


def rm_anova_choice_var_seqlen_incentive(
    df_beh_seq: pd.DataFrame, *, dv_col: str
) -> pd.DataFrame:
    """rmANOVA on behaviour choice_var (log or natural).

    Uses :func:`rm_anova_exact` (exact per-effect Greenhouse-Geisser epsilon, GG p for every effect):
    pingouin's epsilon is wrong for the 3 x 3 seqlen x incentive interaction.
    """
    d = df_beh_seq.dropna(subset=["participant", "seqlen", "incentive", dv_col]).copy()
    if d.empty:
        return pd.DataFrame({"Source": ["error"], "message": ["No rows for rmANOVA."]})
    return rm_anova_exact(d[["participant", "seqlen", "incentive", dv_col]], dv_col, ["seqlen", "incentive"])


_VAR_RED_COMBINED_SUMMARY_COLS: tuple[str, ...] = (
    "contrast",
    "contrast_description",
    "mean_r2",
    "median_r2",
    "sem_r2",
    "n_participants",
    "pooled_r2_all_cells",
    "n_cells_pooled",
    "ttest_per_subj_r2_vs_0_t",
    "ttest_per_subj_r2_vs_0_p_two_sided",
)


def _r2_per_participant_long_to_summary_row(
    long_df: pd.DataFrame,
    *,
    r2_col: str,
    contrast: str,
    contrast_description: str,
    pooled_yo: list[float],
    pooled_yp: list[float],
) -> pd.DataFrame:
    """Build one summary row matching ``_VAR_RED_COMBINED_SUMMARY_COLS`` from a long R² table."""
    empty = pd.DataFrame(columns=list(_VAR_RED_COMBINED_SUMMARY_COLS))
    if long_df.empty or r2_col not in long_df.columns:
        return empty
    arr = pd.to_numeric(long_df[r2_col], errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    mean_r = float(np.nanmean(arr))
    med_r = float(np.median(arr))
    sem_r = (
        float(pd.Series(arr).sem(ddof=1))
        if arr.size > 1
        else (0.0 if arr.size == 1 else float("nan"))
    )
    pr_pool, n_pool = r2_obs_vs_pred_choice_var(
        np.asarray(pooled_yo, dtype=float),
        np.asarray(pooled_yp, dtype=float),
    )
    if arr.size >= 2:
        tt = ttest_1samp(arr, popmean=0.0, nan_policy="omit")
        t_v, p_v = float(tt.statistic), float(tt.pvalue)
    else:
        t_v, p_v = float("nan"), float("nan")
    summary = pd.DataFrame(
        [
            {
                "contrast": contrast,
                "contrast_description": contrast_description,
                "mean_r2": mean_r,
                "median_r2": med_r,
                "sem_r2": sem_r,
                "n_participants": int(arr.size),
                "pooled_r2_all_cells": float(pr_pool),
                "n_cells_pooled": int(n_pool),
                "ttest_per_subj_r2_vs_0_t": t_v,
                "ttest_per_subj_r2_vs_0_p_two_sided": p_v,
            }
        ]
    )
    return summary.reindex(columns=list(_VAR_RED_COMBINED_SUMMARY_COLS))


def choice_var_full_grid_obs_vs_pred_r2_tables(
    m_nat: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Within-participant R² over the full 3×3 (incentive × seqlen) inner-merged grid.

    Requires exactly nine finite ``(choice_var, pred_choice_var)`` pairs per participant at
    incentives ``{-1, 0, +1}`` and three distinct ``seqlen`` values. Rows are sorted by
    ``incentive`` then ``seqlen`` before R².
    """
    need = {"participant", "incentive", "seqlen", "choice_var", "pred_choice_var"}
    empty_long = pd.DataFrame(columns=["participant", "n_points", "r2_obs_vs_pred"])
    empty_sum = pd.DataFrame(columns=list(_VAR_RED_COMBINED_SUMMARY_COLS))
    if m_nat.empty or not need.issubset(m_nat.columns):
        return empty_sum, empty_long

    d = m_nat[list(need)].copy()
    d["participant"] = d["participant"].astype(str)
    d["incentive"] = pd.to_numeric(d["incentive"], errors="coerce")
    d["seqlen"] = pd.to_numeric(d["seqlen"], errors="coerce").astype(int)
    d["choice_var"] = pd.to_numeric(d["choice_var"], errors="coerce")
    d["pred_choice_var"] = pd.to_numeric(d["pred_choice_var"], errors="coerce")
    d = d.dropna(subset=["participant", "incentive", "seqlen", "choice_var", "pred_choice_var"])

    long_rows: list[dict[str, Any]] = []
    pooled_yo: list[float] = []
    pooled_yp: list[float] = []

    for pid, g in d.groupby("participant", sort=False):
        g = g.copy()
        inc_int = g["incentive"].astype(float).round().astype(int)
        if inc_int.nunique() != 3 or g["seqlen"].nunique() != 3 or len(g) != 9:
            continue
        if set(inc_int.unique().tolist()) != {-1, 0, 1}:
            continue
        g2 = g.assign(_inc=inc_int).sort_values(["_inc", "seqlen"])
        yo = g2["choice_var"].to_numpy(dtype=float)
        yp = g2["pred_choice_var"].to_numpy(dtype=float)
        if not (np.all(np.isfinite(yo)) and np.all(np.isfinite(yp))):
            continue
        r2, npts = r2_obs_vs_pred_choice_var(yo, yp)
        if int(npts) != 9 or not np.isfinite(r2):
            continue
        long_rows.append(
            {
                "participant": str(pid),
                "n_points": 9,
                "r2_obs_vs_pred": float(r2),
            }
        )
        pooled_yo.extend(yo.tolist())
        pooled_yp.extend(yp.tolist())

    long_df = pd.DataFrame(long_rows)
    if long_df.empty:
        return empty_sum, empty_long

    desc = (
        "Per participant: nine paired obs vs pred_choice_var cells at incentives -1, 0, +1 "
        "times three seqlen levels (sorted incentive then seqlen). One R2 per person."
    )
    summary = _r2_per_participant_long_to_summary_row(
        long_df,
        r2_col="r2_obs_vs_pred",
        contrast="full_grid_natural",
        contrast_description=desc,
        pooled_yo=pooled_yo,
        pooled_yp=pooled_yp,
    )
    return summary, long_df


def merge_var_red_obs_pred_long(
    obs_long: pd.DataFrame, pred_long: pd.DataFrame
) -> pd.DataFrame:
    """Inner-merge two ``(participant, seqlen, var_red)`` frames into one obs/pred long table.

    ``obs_long`` / ``pred_long`` are outputs of
    :func:`motinf.exploratory.choice_var_seqlen_preprocess.var_red_natural_long_by_seqlen`, applied once to a
    behavioural column and once to a prediction column.
    """
    cols = ["participant", "seqlen", "var_red_obs", "var_red_pred"]
    need = {"participant", "seqlen", "var_red"}
    if (
        obs_long.empty
        or pred_long.empty
        or not (need.issubset(obs_long.columns) and need.issubset(pred_long.columns))
    ):
        return pd.DataFrame(columns=cols)
    o = obs_long[list(need)].rename(columns={"var_red": "var_red_obs"}).copy()
    p = pred_long[list(need)].rename(columns={"var_red": "var_red_pred"}).copy()
    o["participant"] = o["participant"].astype(str)
    p["participant"] = p["participant"].astype(str)
    o["seqlen"] = o["seqlen"].astype(int)
    p["seqlen"] = p["seqlen"].astype(int)
    out = o.merge(p, on=["participant", "seqlen"], how="inner")
    return out[cols]


_VAR_RED_PP_ERROR_EXTRA_COLS: tuple[str, ...] = (
    "mean_mae",
    "median_mae",
    "sem_mae",
    "pooled_mae_all_cells",
    "mean_rmse",
    "median_rmse",
    "sem_rmse",
    "pooled_rmse_all_cells",
)


def _mean_median_sem(arr: np.ndarray) -> tuple[float, float, float]:
    """Mean, median, SEM of finite values in ``arr`` (SEM convention matches R² summaries)."""
    a = arr[np.isfinite(arr)]
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean_v = float(np.mean(a))
    med_v = float(np.median(a))
    sem_v = (
        float(pd.Series(a).sem(ddof=1))
        if a.size > 1
        else 0.0
    )
    return mean_v, med_v, sem_v


def var_red_by_seqlen_r2_per_participant(
    obs_long: pd.DataFrame,
    pred_long: pd.DataFrame,
    *,
    contrast: str,
    contrast_description: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-participant R², MAE, and RMSE of the 3-point (one per seqlen) incentive effect.

    ``obs_long`` / ``pred_long``: the incentive effect (neutral − |incentive| = 1, gain and loss pooled) per
    participant and sequence length, of the data and of the model's predictions
    (:func:`motinf.exploratory.choice_var_seqlen_preprocess.var_red_natural_long_by_seqlen`). Requires exactly three
    finite paired ``seqlen`` points per participant.

    R² at n=3 is fragile — a near-flat participant profile makes SS_tot near zero, so a small
    residual can blow R² up in either direction (see e.g. a participant with R² ~ -25 on this
    metric). MAE and RMSE, in the same units as ``var_red``, don't have that pathology, so the
    summary row reports both alongside R² (**mean/median/sem_mae**, **mean/median/sem_rmse**,
    plus **pooled_mae/rmse_all_cells** computed over every participant's points stacked together,
    mirroring **pooled_r2_all_cells**).
    """
    empty_long = pd.DataFrame(
        columns=[
            "participant",
            "n_points",
            "r2_var_red_obs_vs_pred",
            "mae_var_red_obs_vs_pred",
            "rmse_var_red_obs_vs_pred",
        ]
    )
    empty_sum = pd.DataFrame(
        columns=list(_VAR_RED_COMBINED_SUMMARY_COLS) + list(_VAR_RED_PP_ERROR_EXTRA_COLS)
    )
    m = merge_var_red_obs_pred_long(obs_long, pred_long)
    if m.empty:
        return empty_sum, empty_long
    m = m.dropna(subset=["participant", "seqlen", "var_red_obs", "var_red_pred"])

    long_rows: list[dict[str, Any]] = []
    pooled_yo: list[float] = []
    pooled_yp: list[float] = []

    for pid, g in m.groupby("participant", sort=False):
        g = g.sort_values("seqlen")
        if g["seqlen"].nunique() != 3 or len(g) != 3:
            continue
        yo = g["var_red_obs"].to_numpy(dtype=float)
        yp = g["var_red_pred"].to_numpy(dtype=float)
        if not (np.all(np.isfinite(yo)) and np.all(np.isfinite(yp))):
            continue
        r2, npts = r2_obs_vs_pred_choice_var(yo, yp)
        if int(npts) != 3 or not np.isfinite(r2):
            continue
        resid = yo - yp
        long_rows.append(
            {
                "participant": str(pid),
                "n_points": 3,
                "r2_var_red_obs_vs_pred": float(r2),
                "mae_var_red_obs_vs_pred": float(np.mean(np.abs(resid))),
                "rmse_var_red_obs_vs_pred": float(np.sqrt(np.mean(resid**2))),
            }
        )
        pooled_yo.extend(yo.tolist())
        pooled_yp.extend(yp.tolist())

    long_df = pd.DataFrame(long_rows)
    if long_df.empty:
        return empty_sum, empty_long

    summary = _r2_per_participant_long_to_summary_row(
        long_df,
        r2_col="r2_var_red_obs_vs_pred",
        contrast=contrast,
        contrast_description=contrast_description,
        pooled_yo=pooled_yo,
        pooled_yp=pooled_yp,
    )

    mae_arr = pd.to_numeric(long_df["mae_var_red_obs_vs_pred"], errors="coerce").to_numpy(float)
    rmse_arr = pd.to_numeric(long_df["rmse_var_red_obs_vs_pred"], errors="coerce").to_numpy(float)
    mean_mae, med_mae, sem_mae = _mean_median_sem(mae_arr)
    mean_rmse, med_rmse, sem_rmse = _mean_median_sem(rmse_arr)
    pooled_resid = np.asarray(pooled_yo, dtype=float) - np.asarray(pooled_yp, dtype=float)
    pooled_mae = float(np.mean(np.abs(pooled_resid))) if pooled_resid.size else float("nan")
    pooled_rmse = float(np.sqrt(np.mean(pooled_resid**2))) if pooled_resid.size else float("nan")

    summary["mean_mae"] = mean_mae
    summary["median_mae"] = med_mae
    summary["sem_mae"] = sem_mae
    summary["pooled_mae_all_cells"] = pooled_mae
    summary["mean_rmse"] = mean_rmse
    summary["median_rmse"] = med_rmse
    summary["sem_rmse"] = sem_rmse
    summary["pooled_rmse_all_cells"] = pooled_rmse

    return summary, long_df


def accuracy_by_participant(root: Path, segments: list[str]) -> pd.DataFrame:
    """Per participant, the proportion of correct choices (true category), of optimal choices (consistent with the
    presented evidence, ``optim``) and the accuracy of an ideal observer choosing the sign of the presented evidence
    (``sumllr_noisy``). ``table_stats.accuracy_summary`` averages them per experiment (Results)."""
    parts = []
    for seg in segments:
        d = pd.read_csv(
            root / "data" / "processed" / seg / "data_clean_inf.csv",
            usecols=["participant", "correct", "optim", "sumllr_noisy", "true_cat"],
        )
        d["ideal_observer"] = (np.sign(d["sumllr_noisy"]) == np.sign(d["true_cat"])).astype(float)
        pm = d.groupby("participant")[["correct", "optim", "ideal_observer"]].mean().reset_index()
        parts.append(pm.assign(experiment=seg)[["experiment", "participant", "correct", "optim", "ideal_observer"]])
    return pd.concat(parts, ignore_index=True)


def rt_trial_counts_by_participant(root: Path, segments: list[str]) -> pd.DataFrame:
    """Per retained participant and task: trials before cleaning (``data/interim/<exp>/all_trials_<task>.csv``) and
    after it (``data/processed/<exp>/data_clean_<task>.csv``). ``table_stats.rt_exclusion_summary`` gives the trials
    removed by the trial-level (RT) cut of the pre-registered cleaning (Methods)."""
    parts = []
    for seg in segments:
        for task, label in (("inf", "integration"), ("est", "estimation")):
            clean = pd.read_csv(root / "data" / "processed" / seg / f"data_clean_{task}.csv", usecols=["participant"])
            after = clean["participant"].astype(str).value_counts()
            before = pd.read_csv(root / "data" / "interim" / seg / f"all_trials_{task}.csv", usecols=["participant"])
            before = before["participant"].astype(str).value_counts()
            kept = sorted(after.index)
            parts.append(pd.DataFrame({"experiment": seg, "task": label, "participant": kept,
                                       "n_trials_before": before.reindex(kept).fillna(0).astype(int).to_numpy(),
                                       "n_trials_after": after.reindex(kept).astype(int).to_numpy()}))
    return pd.concat(parts, ignore_index=True)


PSYCHOMETRIC_EDGES = np.array([0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.5, np.inf])  # |Σ log LR| bins (Figure 1f)
PSYCHOMETRIC_GRID = np.linspace(0, 5.5, 120)  # |Σ log LR| at which the probit curves are evaluated


def psychometric_consistency(root: Path, segments: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choices consistent with the displayed evidence as a function of its strength, per incentive (Figure 1f).

    - Binned data: per participant, the proportion of consistent choices in each |summed evidence| bin with at least
      3 trials; then mean ± s.e.m. across participants (``abs_evidence`` = mean |evidence| in the bin).
    - Curves: a probit fit of choice on signed evidence per participant and incentive (intercept b0, slope b1); the
      probability of a consistent choice at |evidence| x, 0.5 [Φ(b0 + b1 x) + Φ(−b0 + b1 x)], averaged over
      participants.

    Returns (bins, curves).
    """
    inf = []
    for e in segments:
        t = pd.read_csv(root / "data" / "processed" / e / "data_clean_inf.csv",
                        usecols=["participant", "incentive", "sumllr_noisy", "choice"])
        t["participant"] = t["participant"].astype(str) + "__" + e
        inf.append(t)
    inf = pd.concat(inf, ignore_index=True).dropna()
    inf["absev"] = inf["sumllr_noisy"].abs()
    inf["consistent"] = (np.sign(inf["choice"]) == np.sign(inf["sumllr_noisy"])).astype(float)
    inf["bin"] = pd.cut(inf["absev"], PSYCHOMETRIC_EDGES, labels=False, right=False)
    bins, curves = [], []
    for inc in (0.0, -1.0, 1.0):
        sub = inf[inf["incentive"] == inc]
        pp = sub.groupby(["participant", "bin"]).agg(p=("consistent", "mean"), x=("absev", "mean"), n=("consistent", "size"))
        pp = pp[pp["n"] >= 3].reset_index()
        g = pp.groupby("bin").agg(m=("p", "mean"), s=("p", lambda v: v.std(ddof=1) / np.sqrt(len(v))), x=("x", "mean"),
                                  n=("p", "size"))
        bins.append(pd.DataFrame({"incentive": inc, "bin": g.index, "abs_evidence": g["x"].to_numpy(),
                                  "p_consistent_mean": g["m"].to_numpy(), "p_consistent_sem": g["s"].to_numpy(),
                                  "n_participants": g["n"].to_numpy()}))
        per = []
        for _, q in sub.groupby("participant"):
            res = sm.Probit((q["choice"].to_numpy() > 0).astype(float), sm.add_constant(q["sumllr_noisy"].to_numpy())).fit(disp=False)
            b0, b1 = res.params
            per.append(0.5 * (norm.cdf(b0 + b1 * PSYCHOMETRIC_GRID) + norm.cdf(-b0 + b1 * PSYCHOMETRIC_GRID)))
        curves.append(pd.DataFrame({"incentive": inc, "abs_evidence": PSYCHOMETRIC_GRID,
                                    "p_consistent": np.mean(per, axis=0), "n_participants": len(per)}))
    return pd.concat(bins, ignore_index=True), pd.concat(curves, ignore_index=True)


def run_behavior_stats(root: Path, exp_name: str) -> None:
    """Behaviour statistics outside the pre-registered pipeline, for one experiment or the pooled
    sample (``results/exploratory/<exp>/behavior/stats/behav_extra/params/``):

    - ``model_pred_per_seqlen_simple_inf.csv``: choice variability per participant, incentive and
      sequence length predicted by the per-condition inference-noise fits (Figure 2d);

    and, for the pooled sample only (its tables hold one row per experiment where the paper reports experiments):

    - ``accuracy_by_participant.csv``: accuracy, optimal choices and ideal-observer accuracy per participant;
    - ``rt_trial_counts_by_participant.csv``: trials of each retained participant before and after cleaning;
    - ``psychometric_consistency_{bins,curves}.csv``: choices consistent with the evidence against its strength,
      binned data and the mean of the participants' probit fits (Figure 1f).

    The statistics on these tables (natural-scale choice-variability rmANOVA, R² of the inference-only model, mean
    accuracies, response-time exclusions) are computed by the ``tables`` stage (``motinf.exploratory.table_stats``).
    """
    cohort_segs = cohort_subexperiments(exp_name)
    validate_prereg_inputs_exist(root, cohort_segs)
    params_dir = behav_extra_dir(root, exp_name) / "params"
    params_dir.mkdir(parents=True, exist_ok=True)

    df_trials = merge_inf_trials_for_choice_var(root, cohort_segs)
    df_sd_inf_cell = merge_sd_inf_long(root, cohort_segs)[["participant", "incentive", "sd_inf"]]
    df_pred_per = predict_inf_choicevar_per_seqlen(df_trials, df_sd_inf_cell)
    df_pred_per.to_csv(params_dir / "model_pred_per_seqlen_simple_inf.csv", index=False)
    if len(cohort_segs) > 1:
        rt_trial_counts_by_participant(root, cohort_segs).to_csv(params_dir / "rt_trial_counts_by_participant.csv",
                                                                 index=False)
        accuracy_by_participant(root, cohort_segs).to_csv(params_dir / "accuracy_by_participant.csv", index=False)
        bins, curves = psychometric_consistency(root, cohort_segs)
        bins.to_csv(params_dir / "psychometric_consistency_bins.csv", index=False)
        curves.to_csv(params_dir / "psychometric_consistency_curves.csv", index=False)
    print(f"[ok] behaviour stats -> {params_dir}")
