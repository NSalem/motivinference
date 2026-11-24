"""Statistics computed from participant-level tables only (the ``tables`` stage of ``run_pipeline.py``).

Every function here reads tables that the ``stats`` stage wrote to ``results/`` (never trial data or fit files) and
writes its output next to them, at the paths the rest of the code and ``PAPER_MAP.md`` use. With the code repository
alone, ``python run_pipeline.py --stage tables figures verify`` therefore recomputes these statistics,
``paper_tables.json`` and every figure.

- :func:`behavior_tables` (per experiment and pooled): choice-variability rmANOVA on the natural scale
  (``behav_extra/params``); pooled only, the R² of the inference-only model, accuracy against the ideal observer and
  trials removed by the response-time cut.
- :func:`full_model_tables` (pooled): δ against 0, variance partition, correlation of the δ, robust regressions of
  Supplementary Table 4, the correlations of Supplementary Figure 6, and the prediction error of the full model and
  of the knockouts.
- :func:`inference_model_tables` (pooled): correlation of the incentive effects between tasks, group Bayesian model
  selection of the single-noise and single-modulation models, and the prediction error of the single-modulation
  models.
- :func:`full_vs_inference_only_tables`: group Bayesian model selection of the full model against the inference-only
  models.

Tables are read with ``float_precision="round_trip"``, so every number equals the one the stats stage held in memory.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_1samp

from motinf.exploratory.behav_stats_paths import behav_extra_dir
from motinf.exploratory.behavior_stats import (
    choice_var_full_grid_obs_vs_pred_r2_tables,
    rm_anova_choice_var_seqlen_incentive,
    var_red_by_seqlen_r2_per_participant,
)
from motinf.exploratory.choice_var_seqlen_preprocess import (
    merge_and_preprocess_choice_var_beh_pred,
    var_red_natural_long_by_seqlen,
)
from motinf.exploratory.cohorts import (
    cohort_subexperiments,
    merge_behavior_mean_inc,
    merge_behavior_seqlen,
    merge_sd_inf_long,
)
from motinf.exploratory.full_model_comparison import group_bmc_from_aicc
from motinf.exploratory.full_model_results import CHANNELS, compute_motiv_correlation_table
from motinf.exploratory.rlm_regression import fit_rlm_block
from motinf.exploratory.single_modulation_fits import SINGLE_MODULATION_MODELS
from motinf.exploratory.stats import group_bmc, holm
from motinf.exploratory.variance_partition import log_decomposition_partition


def read(path: Path) -> pd.DataFrame:
    """A table written by the stats stage, numbers exactly as written."""
    return pd.read_csv(path, float_precision="round_trip")


def _behaviour_seqlen(root: Path, segments: list[str]) -> pd.DataFrame:
    df = merge_behavior_seqlen(root, segments)
    df["participant"] = df["participant"].astype(str)
    df["incentive"] = df["incentive"].astype(float)
    df["seqlen"] = df["seqlen"].astype(int)
    return df


# --------------------------------------------------------------------------- #
# Behaviour
# --------------------------------------------------------------------------- #


def accuracy_summary(per_participant: pd.DataFrame) -> pd.DataFrame:
    """Per experiment, means over participants of the proportions of correct, optimal and ideal-observer choices
    (``accuracy_by_participant.csv``, written by ``behavior_stats.accuracy_by_participant``)."""
    rows = []
    for seg, pm in per_participant.groupby("experiment", sort=False):
        pm = pm[["correct", "optim", "ideal_observer"]]
        rows.append({"experiment": seg, "n_participants": len(pm), **{f"{c}_mean": float(pm[c].mean()) for c in pm}})
    return pd.DataFrame(rows)


def rt_exclusion_summary(per_participant: pd.DataFrame) -> pd.DataFrame:
    """Per experiment and task, trials of retained participants before cleaning and how many the response-time cut
    removed (``rt_trial_counts_by_participant.csv``, written by ``behavior_stats.rt_trial_counts_by_participant``)."""
    rows = []
    for (seg, task), g in per_participant.groupby(["experiment", "task"], sort=False):
        n_before, n_after = int(g["n_trials_before"].sum()), int(g["n_trials_after"].sum())
        n_removed = n_before - n_after
        rows.append({"experiment": seg, "task": task, "n_participants": len(g), "n_trials": n_before,
                     "n_trials_removed": n_removed, "pct_removed": 100.0 * n_removed / n_before})
    return pd.DataFrame(rows)


def behavior_tables(root: Path, exp_name: str) -> None:
    """Natural-scale choice-variability rmANOVA (participants with extreme values excluded), for one experiment or the
    pooled sample; for the pooled sample also the R² of the inference-only model, accuracy and response-time
    exclusions (one row per experiment where the paper reports them per experiment)."""
    segments = cohort_subexperiments(exp_name)
    params_dir = behav_extra_dir(root, exp_name) / "params"
    df_beh_seq = _behaviour_seqlen(root, segments)
    df_pred = read(params_dir / "model_pred_per_seqlen_simple_inf.csv")
    m_nat = merge_and_preprocess_choice_var_beh_pred(df_beh_seq, df_pred)
    rm_anova_choice_var_seqlen_incentive(m_nat, dv_col="choice_var").to_csv(
        params_dir / "rm_anova_choice_var_natural_seqlen_x_incentive.csv", index=False)
    if len(segments) == 1:
        return
    r2, _ = choice_var_full_grid_obs_vs_pred_r2_tables(m_nat)
    r2.to_csv(params_dir / "r2_choice_var_full_grid_natural_obs_vs_simple_inf.csv", index=False)
    rt_exclusion_summary(read(params_dir / "rt_trial_counts_by_participant.csv")).to_csv(
        params_dir / "rt_trial_exclusions.csv", index=False)
    accuracy_summary(read(params_dir / "accuracy_by_participant.csv")).to_csv(
        params_dir / "accuracy_vs_ideal_observer.csv", index=False)


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #


def motiv_ttests(df_pars: pd.DataFrame) -> pd.DataFrame:
    """One-sample t-tests of each incentive modulation ``motiv_X`` against 0, Holm over the three noise sources
    (Figure 3a stars, Results)."""
    rows, p_raw = [], []
    for ch in CHANNELS:
        m = df_pars[f"motiv_{ch}"].astype(float).to_numpy()
        m = m[np.isfinite(m)]
        t_res = ttest_1samp(m, 0.0)
        d = float(np.mean(m) / np.std(m, ddof=1)) if np.std(m, ddof=1) > 0 else np.nan
        rows.append({"test": f"motiv_{ch} vs 0", "n": int(len(m)), "t": float(t_res.statistic),
                     "p_raw": float(t_res.pvalue), "cohen_d": d, "mean": float(np.mean(m))})
        p_raw.append(float(t_res.pvalue))
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(p_raw)
    return out


def channel_partition(df_pars: pd.DataFrame, *, n_boot: int = 5000, seed: int = 0):
    """Log decomposition of the channel variances ``total_var_X`` / ``total_var_X_v`` (see
    :func:`motinf.exploratory.variance_partition.log_decomposition_partition`): ``bar[k, j]`` is channel
    ``CHANNELS[k]`` at |incentive| ``j`` as a fraction of the baseline total, with participant-bootstrap replicates and
    the mask of participants used."""
    comp = np.stack(
        [df_pars[[f"total_var_{ch}", f"total_var_{ch}_v"]].to_numpy(dtype=float) for ch in CHANNELS], axis=1)
    return log_decomposition_partition(comp, ref=0, n_boot=n_boot, seed=seed)


def variance_partition_summary(df_pars: pd.DataFrame) -> pd.DataFrame:
    """Group variance partition by channel and |incentive| (``channel_partition``), plus the ``total`` row, with
    participant-bootstrap 95% CIs. ``prop_baseline_total`` is a fraction of the baseline total (Figure 3b, Results)."""
    bar, boot, keep = channel_partition(df_pars)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    tot_lo, tot_hi = np.percentile(boot.sum(axis=1), [2.5, 97.5], axis=0)
    rows = []
    for k, ch in enumerate((*CHANNELS, "total")):
        for j in (0, 1):
            is_tot = ch == "total"
            rows.append({"channel": ch, "incentive_abs": j,
                         "prop_baseline_total": float(bar[:, j].sum() if is_tot else bar[k, j]),
                         "ci_lo": float(tot_lo[j] if is_tot else lo[k, j]),
                         "ci_hi": float(tot_hi[j] if is_tot else hi[k, j]), "n": int(keep.sum())})
    return pd.DataFrame(rows)


def variance_partition_params(df_pars: pd.DataFrame) -> pd.DataFrame:
    """Per participant, channel and |incentive|: predicted variance, also as a share of the baseline total."""
    rows = []
    base_total = df_pars.set_index("participant")["total_var"].astype(float)
    for _, row in df_pars.iterrows():
        pid = str(row["participant"])
        bl = float(base_total.loc[pid])
        for ch in CHANNELS:
            for inc_abs, suffix in [(0, ""), (1, "_v")]:
                tv = float(row[f"total_var_{ch}{suffix}"])
                rows.append({"participant": pid, "channel": ch, "incentive_abs": inc_abs, "total_var": tv,
                             "total_var_prop_baseline": tv / bl if bl > 0 else np.nan})
    return pd.DataFrame(rows)


def param_behavior_rlm(df_main: pd.DataFrame, df_diff: pd.DataFrame) -> pd.DataFrame:
    """Robust regressions of the choice-variance midpoint and slope (data and full-model predictions) on the three
    noise parameters (``main``) and of their incentive effects on the three modulations (``diff``): Huber RLM on
    z-scored predictors with its weighted R² (:func:`motinf.exploratory.rlm_regression.fit_rlm_block`); Supplementary
    Table 4, Supplementary Figure 6."""
    blocks = [("main", df_main, [f"sd_{ch}" for ch in CHANNELS], ["intercept", "slope", "intercept_mod", "slope_mod"]),
              ("diff", df_diff, [f"motiv_{ch}" for ch in CHANNELS],
               ["intercept_diff", "slope_diff", "intercept_mod_diff", "slope_mod_diff"])]
    rows = []
    for block, df_block, x_cols, y_cols in blocks:
        for y in y_cols:
            fit = fit_rlm_block(df_block, y, x_cols)
            rows += [{"block": block, "y": y, **row, "n": fit["n"], "r2_full": fit["r2_full"]} for row in fit["rows"]]
    return pd.DataFrame(rows)


def r2_around_line(x: np.ndarray, y: np.ndarray, b: float, a: float) -> float:
    """R² of the data ``y`` around the fixed line ``a + b·x`` (not refitted to the data, so it can be negative)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < 2:
        return float("nan")
    ss_res = float(np.sum((y - (a + b * x)) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def param_behavior_correlations(df_main: pd.DataFrame, df_diff: pd.DataFrame) -> pd.DataFrame:
    """Supplementary Figure 6a, c: for each noise source, the Pearson correlation (and P) of the data's choice-variance
    midpoint and slope with the source's parameter (``main``: σ; ``diff``: δ against the incentive effects), and the
    R² of the data around the line fitted to the full model's predictions."""
    blocks = [("main", df_main, "sd_{}", [("intercept", "intercept_mod"), ("slope", "slope_mod")]),
              ("diff", df_diff, "motiv_{}", [("intercept_diff", "intercept_mod_diff"), ("slope_diff", "slope_mod_diff")])]
    rows = []
    for block, df, x_name, ys in blocks:
        for y, y_model in ys:
            for ch in CHANNELS:
                x = x_name.format(ch)
                d = df[[x, y]].dropna()
                r, p = pearsonr(d[x], d[y])
                dm = df[[x, y, y_model]].dropna()
                r2 = float("nan")
                if dm.shape[0] >= 3 and float(dm[x].std()) > 0:
                    xv = dm[x].to_numpy(float)
                    b, a = np.polyfit(xv, dm[y_model].to_numpy(float), 1)
                    r2 = r2_around_line(xv, dm[y].to_numpy(float), b, a)
                rows.append({"block": block, "y": y, "x": x, "pearson_r": float(r), "p": float(p), "model_r2": r2,
                             "n": int(len(d))})
    return pd.DataFrame(rows)


KNOCKOUT_MODEL_IDS = ("full", "ko_ang", "ko_inf", "ko_sel")


def incentive_effect_prediction_error(df_beh_seq: pd.DataFrame, predictions: pd.DataFrame, models: tuple[str, ...],
                                      *, model_col: str, predictions_of: str,
                                      report_label: str | None = None) -> pd.DataFrame:
    """Per participant and model, fit of the |incentive| effect on choice variability over the three sequence lengths
    (R², mean absolute error) by the model's predicted choice variability (``predictions``: participant, incentive,
    sequence length, ``pred_choice_var``; the model in ``model_col``)."""
    data = df_beh_seq[["participant", "incentive", "seqlen", "choice_var", "choice_var_log"]]
    parts = []
    for mid in models:
        agg = predictions[predictions[model_col] == mid].drop(columns=[model_col]).reset_index(drop=True)
        m_nat = merge_and_preprocess_choice_var_beh_pred(
            data, agg, report_label=f"{report_label} {mid}" if report_label else None)
        df_mid, _ = var_red_by_seqlen_r2_per_participant(
            var_red_natural_long_by_seqlen(m_nat, "choice_var"),
            var_red_natural_long_by_seqlen(m_nat, "pred_choice_var"),
            contrast=mid,
            contrast_description=("Per participant: 3-point var_red (neutral - |incentive|=1, gain/loss data "
                                  f"averaged) at each seqlen, behaviour vs {predictions_of} predictions."),
        )
        df_mid.insert(0, "model_id", mid)
        parts.append(df_mid)
    return pd.concat(parts, ignore_index=True)


def knockout_prediction_error(df_beh_seq: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """The full model and each knockout (Figure 3d), from ``knockout_predictions.csv``."""
    return incentive_effect_prediction_error(df_beh_seq, predictions, KNOCKOUT_MODEL_IDS, model_col="model_id",
                                             predictions_of="knockout model",
                                             report_label="full_model_stats abs grid/var_red R2")


def single_modulation_prediction_error(df_beh_seq: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Each single-modulation model (Supplementary Figure 4c), from ``pred_choice_var_single_modulation.csv``."""
    return incentive_effect_prediction_error(df_beh_seq, predictions, SINGLE_MODULATION_MODELS, model_col="model",
                                             predictions_of="incentive-effects model")


def full_model_tables(root: Path, exp_name: str = "merged") -> None:
    """Tests on the full-model parameters and predictions (``modeling/stats/full_model_stats``)."""
    out = root / "results" / "exploratory" / exp_name / "modeling" / "stats" / "full_model_stats"
    params, corr = out / "params", out / "correlations"
    df_pars = read(params / "params_wide.csv")
    df_pars["participant"] = df_pars["participant"].astype(str)
    motiv_ttests(df_pars).to_csv(params / "ttests_motiv_vs0.csv", index=False)
    knockout_prediction_error(_behaviour_seqlen(root, cohort_subexperiments(exp_name)),
                              read(out / "knockout" / "knockout_predictions.csv")).to_csv(
        params / "r2_var_red_natural_abs_obs_vs_knockout_model.csv", index=False)
    variance_partition_params(df_pars).to_csv(params / "variance_partition_params.csv", index=False)
    variance_partition_summary(df_pars).to_csv(params / "variance_partition_summary.csv", index=False)
    compute_motiv_correlation_table(df_pars).to_csv(corr / "corr_motiv_params.csv", index=False)
    main, diff = read(corr / "param_behavior_main.csv"), read(corr / "param_behavior_diff.csv")
    param_behavior_rlm(main, diff).to_csv(corr / "rlm_coefs.csv", index=False)
    param_behavior_correlations(main, diff).to_csv(corr / "param_behavior_correlations.csv", index=False)


# --------------------------------------------------------------------------- #
# Inference-only and single-modulation models
# --------------------------------------------------------------------------- #


def _effects_table(df_sd_inf: pd.DataFrame, df_estim: pd.DataFrame) -> pd.DataFrame:
    """Per participant, gain - neutral and loss - neutral of log sd_inf and of log reproduction error SD."""
    pivot_sd_inf = df_sd_inf.pivot(index="participant", columns="incentive", values="sd_inf_log")
    pivot_estim = df_estim.pivot(index="participant", columns="incentive", values="estim_sd_log")
    rows = []
    for pid in sorted(set(pivot_sd_inf.index) | set(pivot_estim.index)):
        row: dict[str, Any] = {"participant": str(pid)}
        for pivot, key in [(pivot_sd_inf, "sd_inf"), (pivot_estim, "estim_sd")]:
            if pid in pivot.index:
                ser = pivot.loc[pid]
                v_g, v_n, v_l = ser.get(1.0, np.nan), ser.get(0.0, np.nan), ser.get(-1.0, np.nan)
            else:
                v_g = v_n = v_l = np.nan
            row[f"gain_log_{key}"] = float(v_g) - float(v_n) if np.isfinite(v_g) and np.isfinite(v_n) else np.nan
            row[f"loss_log_{key}"] = float(v_l) - float(v_n) if np.isfinite(v_l) and np.isfinite(v_n) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _corr_row(label: str, x: pd.Series, y: pd.Series) -> tuple[dict[str, Any], float]:
    df = pd.concat([x, y], axis=1).dropna()
    if df.shape[0] < 3:
        return {"test": label, "n": int(df.shape[0]), "r": np.nan, "p_raw": np.nan}, float("nan")
    r, p = pearsonr(df.iloc[:, 0], df.iloc[:, 1])
    return {"test": label, "n": int(df.shape[0]), "r": float(r), "p_raw": float(p)}, float(p)


def corr_between_tasks(root: Path, segments: list[str]) -> pd.DataFrame:
    """Correlation of the incentive effects on log inference noise and on the log reproduction error SD, gain and
    loss, Holm-corrected (Results)."""
    df_sd_inf = merge_sd_inf_long(root, segments)
    df_m = merge_behavior_mean_inc(root, segments)
    df_estim = df_m[["participant", "incentive", "estim_sd"]].copy()
    df_estim["participant"] = df_estim["participant"].astype(str)
    df_estim["incentive"] = df_estim["incentive"].astype(float)
    df_estim = df_estim.dropna(subset=["estim_sd"]).copy()
    df_estim["estim_sd_log"] = np.log(df_estim["estim_sd"])
    eff = _effects_table(df_sd_inf, df_estim)
    row_g, p_g = _corr_row("between: gain log(estim_sd) vs gain log(sd_inf)", eff["gain_log_estim_sd"],
                           eff["gain_log_sd_inf"])
    row_l, p_l = _corr_row("between: loss log(estim_sd) vs loss log(sd_inf)", eff["loss_log_estim_sd"],
                           eff["loss_log_sd_inf"])
    out = pd.DataFrame([row_g, row_l])
    out["p_holm"] = holm([p_g, p_l])
    return out


def _bmc_metrics(aicc: np.ndarray, n_models: int) -> dict[str, np.ndarray]:
    res = group_bmc(-0.5 * aicc)
    out = {"frequency_mean": res["frequency_mean"], "frequency_var": res["frequency_var"],
           "xp": res["exceedance_probability"], "pxp": res["protected_exceedance_probability"]}
    if any(len(v) != n_models for v in out.values()):
        raise ValueError("GroupBMC result length mismatch with model count.")
    return out


def _aicc_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Models x participants AICc in the order of the table (as the stats stage wrote it)."""
    models = list(dict.fromkeys(df["model"].astype(str)))
    participants = np.asarray(list(dict.fromkeys(df["participant"].astype(str))), dtype=object)
    wide = df.pivot(index="model", columns="participant", values="aicc").reindex(index=models, columns=participants)
    return wide.to_numpy(dtype=float), models, participants


def _drop_incomplete(label: str, aicc: np.ndarray, participants: np.ndarray) -> np.ndarray:
    keep = np.all(np.isfinite(aicc), axis=0)
    if (~keep).any():
        warnings.warn(f"{label}: excluded {int((~keep).sum())}/{len(participants)} participant(s) with incomplete "
                      f"AICc across models: {[str(p) for p in participants[~keep]]}", UserWarning, stacklevel=2)
    return aicc[:, keep]


def group_bmc_noise_per_incentive(aicc_long: pd.DataFrame) -> pd.DataFrame:
    """Group BMC of the single-noise models in each condition (Figure 2b), from ``aicc_noise_per_cond_by_participant.csv``."""
    rows = []
    for inc in dict.fromkeys(aicc_long["incentive"].astype(float)):
        mat, models, participants = _aicc_matrix(aicc_long[aicc_long["incentive"].astype(float) == inc])
        mat = _drop_incomplete(f"GroupBMC merged per-condition (incentive={inc})", mat, participants)
        if mat.shape[1] == 0:
            continue
        met = _bmc_metrics(mat, len(models))
        rows += [{"incentive": float(inc), "model": m, "n": int(mat.shape[1]),
                  **{k: float(met[k][mi]) for k in ("frequency_mean", "frequency_var", "xp", "pxp")}}
                 for mi, m in enumerate(models)]
    return pd.DataFrame(rows)


def group_bmc_incentive_effects(aicc_long: pd.DataFrame) -> pd.DataFrame:
    """Group BMC of the single-modulation models (Supplementary Figure 4a), from ``aicc_inc_effects_by_participant.csv``."""
    mat, models, participants = _aicc_matrix(aicc_long)
    mat = _drop_incomplete("GroupBMC merged incentive-effects", mat, participants)
    met = _bmc_metrics(mat, len(models))
    return pd.DataFrame([{"model": m, "n": int(mat.shape[1]),
                          **{k: float(met[k][mi]) for k in ("frequency_mean", "frequency_var", "xp", "pxp")}}
                         for mi, m in enumerate(models)])


def inference_model_tables(root: Path, exp_name: str = "merged") -> None:
    """Tests of the inference-only and single-modulation models (``modeling/stats/model_fit_stats``)."""
    segments = cohort_subexperiments(exp_name)
    out = root / "results" / "exploratory" / exp_name / "modeling" / "stats" / "model_fit_stats"
    params = out / "params"
    corr_between_tasks(root, segments).to_csv(out / "correlations" / "corr_between_tasks.csv", index=False)
    single_modulation_prediction_error(_behaviour_seqlen(root, segments),
                                       read(params / "pred_choice_var_single_modulation.csv")).to_csv(
        params / "r2_var_red_natural_abs_obs_vs_inc_eff_model.csv", index=False)
    if len(segments) > 1:  # per experiment, the pre-registered model statistics give these
        group_bmc_noise_per_incentive(read(params / "aicc_noise_per_cond_by_participant.csv")).to_csv(
            params / "groupbmc_noise_per_incentive.csv", index=False)
        group_bmc_incentive_effects(read(params / "aicc_inc_effects_by_participant.csv")).to_csv(
            params / "groupbmc_incentive_effects.csv", index=False)


# --------------------------------------------------------------------------- #
# Full model against the inference-only models
# --------------------------------------------------------------------------- #

FULL_VS_INF_MODELS = ["full", "inf_v_prereg", "noise_per_cond_inf"]
FULL_VS_INF_AICC = {"full": "aicc_bads_full", "inf_v_prereg": "aicc_inf_v_prereg",  # model -> AICc column
                    "noise_per_cond_inf": "aicc_noise_per_cond_inf"}


def full_vs_inference_only_tables(root: Path) -> None:
    """Group BMC of the full model against the two inference-only models (Results), with the full model scored at its
    best BADS (MAP) solution, from ``aicc_full_vs_infv_vs_noise_per_cond_by_participant.csv``."""
    out = root / "results" / "exploratory" / "merged" / "modeling" / "stats" / "full_vs_inference_only"
    df = read(out / "aicc_full_vs_infv_vs_noise_per_cond_by_participant.csv")
    group_bmc_from_aicc({m: df[c].to_numpy(dtype=float) for m, c in FULL_VS_INF_AICC.items()},
                        FULL_VS_INF_MODELS).to_csv(out / "groupbmc_full_vs_infv_vs_noise_per_cond.csv", index=False)


def run_table_stats(root: Path) -> None:
    for exp_name in ("exp1", "exp2", "merged"):
        behavior_tables(root, exp_name)
    inference_model_tables(root, "merged")
    full_model_tables(root, "merged")
    full_vs_inference_only_tables(root)
