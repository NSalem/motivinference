"""
Reusable data cleaning and quality-check routines.

Factored from a legacy quality-check script (external / historical reference),
exposed here as a reusable function for different data sets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import norm
import statsmodels.api as sm

from .stats import sliding_psychometric


# ---------------------------- Configuration ---------------------------- #

@dataclass
class QCConfig:
    """
    Configuration for the quality-check and exclusion procedure.

    Thresholds match the legacy script this module was refactored from.
    """

    # Output
    outdir: str = "Results_QC"

    # Trial-level RT exclusion (both tasks)
    rt_fast_s: float = 0.2
    rt_slow_s: float = 10
    use_trial_rt_exclusion: bool = True

    # Categorization: drop trials whose choice is not in {-1, +1} (e.g. missed response).
    # When True, those trials also count toward the participant-level bad-trial fraction
    # (together with RT too fast/slow) compared to slowfast_frac_thresh.
    use_trial_valid_choice_exclusion: bool = True

    # Absolute RT SD threshold (participant-level), in seconds
    rt_sd_min_s: float = 0.100  # 100 ms

    # Repetition threshold (categorization)
    same_dir_thresh: float = 0.90

    # MAD thresholds
    z_mad: float = 4

    # Participant-level criteria switches
    # Either-task criteria
    use_slowfast_exclusion: bool = True
    use_rt_sd_low_exclusion: bool = True

    use_bias_mad_outlier_exclusion: bool = True
    use_noise_mad_outlier_exclusion: bool = True
    use_log_noise_for_mad: bool = True

    # Categorization criteria
    use_acc_above_optimal: bool = True  # binomial vs optimal (any incentive)
    use_easy_top_not_above_chance_exclusion: bool = True
    use_slope_leq_0_exclusion: bool = True
    use_same_choice_repetition_exclusion: bool = True

    # ---------------- New participant-level exclusion criteria ---------------- #
    # Bad-trial fraction threshold (categorization: invalid choice and/or RT out of range;
    # estimation: RT out of range). Participant excluded if >= this on either task.
    slowfast_frac_thresh: float = 0.20

    # "Easy" subset for the top-10% trials criterion (categorization task)
    easy_top_frac: float = 0.10  # top fraction by abs evidence
    easy_quantile: float = 0.90  # |evidence| threshold = this quantile of abs(evidence) per participant

    # Estimation criteria
    use_estim_sd_dev_noisy_exclusion: bool = True
    estim_sd_dev_noisy_max_deg: float = 30.0
    use_estim_sd_response_low_exclusion: bool = True
    estim_sd_response_min_deg: float = 10.0

    # Incentive levels
    inc_levels: tuple = (0, 1, -1)
    inc_names: Optional[Dict[int, str]] = None

    # Other
    alpha_cat: float = 0.05

    # Plot layout
    n_cols_panels: int = 6

    # Whether to produce and save figures
    make_plots: bool = True

    def __post_init__(self) -> None:
        if self.inc_names is None:
            self.inc_names = {0: "0", 1: "g", -1: "l"}


# ----------------------------- Helpers -------------------------------- #

def wrap_err_deg(resp_deg: np.ndarray, true_deg: np.ndarray) -> np.ndarray:
    """Wrap signed error to [-90, 90] for semicircle (0..180)."""
    return (resp_deg - true_deg + 90) % 180 - 90


def robust_z_series(s: pd.Series) -> pd.Series:
    """MAD-based robust z, preserving index; ignores NaNs."""
    s = s.astype(float)
    med = np.nanmedian(s)
    mad = np.nanmedian(np.abs(s - med))
    if mad == 0 or np.isnan(mad):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return 0.6745 * (s - med) / mad


def fit_probit_glm(x, y):
    """
    Binomial probit GLM : P(y=1|x) = Phi(beta0 + beta1 x) = Phi((x - mu) / sigma)
    with mu = -beta0/beta1 and sigma = 1/beta1.

    Returns (mu, sigma) in evidence units, or (nan, nan) if not identifiable
    (too few points, no outcome variation, non-positive slope, or fit failure).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2 or len(np.unique(y)) < 2:
        return np.nan, np.nan
    try:
        X = sm.add_constant(x)
        model = sm.GLM(y, X, family=sm.families.Binomial(link=sm.families.links.Probit()))
        res = model.fit(disp=0)
    except Exception:
        return np.nan, np.nan

    beta0 = float(res.params[0])
    beta1 = float(res.params[1])
    if not np.isfinite(beta0) or not np.isfinite(beta1) or beta1 <= 0:
        return np.nan, np.nan

    sigma = 1.0 / beta1
    mu = -beta0 / beta1
    return float(mu), float(sigma)


# ----------------------- Main reusable function ----------------------- #

def quality_check_exclusion_separate_tasks(
    df_cat: Union[pd.DataFrame, str, os.PathLike],
    df_est: Union[pd.DataFrame, str, os.PathLike],
    out_cat_path: Optional[Union[str, os.PathLike]] = None,
    out_est_path: Optional[Union[str, os.PathLike]] = None,
    config: Optional[QCConfig] = None,
) -> Dict[str, Any]:
    """
    Run the QC + exclusion pipeline starting from separate
    categorization and estimation tables (as produced by aggregation).

    Parameters
    ----------
    df_cat, df_est
        Either pandas DataFrames or paths to CSV files containing the
        categorization / estimation trials.
    out_cat_path, out_est_path
        Optional output paths for the cleaned categorization and
        estimation CSVs. If omitted, defaults to
        `data_clean_cat.csv` / `data_clean_est.csv` in the current
        working directory. The prereg entrypoint
        (`motinf.prereg.cleaning.run_cleaning_prereg`) passes
        `data_clean_inf.csv` and `data_clean_est.csv` under
        `data/processed/<exp_name>/`.

    The participant-level QC logic matches `quality_check_exclusion`:
    we compute QC summaries for each task, derive participant flags,
    form a joint exclusion mask, and then remove excluded participants
    from both tasks' data.

    For categorization, when ``use_trial_valid_choice_exclusion`` is True (default),
    trials with ``choice`` not in ``{-1, 1}`` are dropped and counted as bad trials
    for the participant-level bad-trial fraction (with RT too fast/slow).
    """

    if config is None:
        config = QCConfig()

    os.makedirs(config.outdir, exist_ok=True)

    # Allow passing either DataFrames or file paths
    if not isinstance(df_cat, pd.DataFrame):
        df_cat = pd.read_csv(df_cat)
    if not isinstance(df_est, pd.DataFrame):
        df_est = pd.read_csv(df_est)

    if out_cat_path is None:
        out_cat_path = "data_clean_cat.csv"
    if out_est_path is None:
        out_est_path = "data_clean_est.csv"

    rt_fast = config.rt_fast_s
    rt_slow = config.rt_slow_s

    # ------------------------ Trial-level RT exclusion ------------------------ #
    df_categ = df_cat.copy()
    df_estim = df_est.copy()

    # Participant bad-trial fraction for categorization (computed on raw pre-trial-exclusion data).
    # Bad = RT too fast/slow when RT is present, and optionally choice not in {-1, +1}.
    rt_c = pd.to_numeric(df_categ["rt"], errors="coerce")
    ch_c = pd.to_numeric(df_categ["choice"], errors="coerce")
    choice_ok = ch_c.notna() & ch_c.isin([-1.0, 1.0])
    invalid_choice = ~choice_ok
    is_slowfast_rt = rt_c.notna() & ((rt_c < rt_fast) | (rt_c > rt_slow))
    if config.use_trial_valid_choice_exclusion:
        is_bad_categ = invalid_choice | is_slowfast_rt
    else:
        is_bad_categ = is_slowfast_rt
    df_categ_sf = pd.DataFrame(
        {
            "participant": df_categ["participant"],
            "rt_valid": rt_c.notna(),
            "is_bad": is_bad_categ,
        }
    )
    if config.use_trial_valid_choice_exclusion:
        sf_c = (
            df_categ_sf.groupby("participant", observed=True)
            .agg(n_categ_trials=("is_bad", "size"), n_categ_bad=("is_bad", "sum"))
            .reset_index()
        )
    else:
        sf_c = (
            df_categ_sf.loc[df_categ_sf["rt_valid"]]
            .groupby("participant", observed=True)
            .agg(n_categ_trials=("is_bad", "size"), n_categ_bad=("is_bad", "sum"))
            .reset_index()
        )
    sf_c["frac_bad_categ"] = sf_c["n_categ_bad"] / sf_c["n_categ_trials"].replace(0, np.nan)

    rt_e = pd.to_numeric(df_estim["rt"], errors="coerce")
    df_estim_sf = pd.DataFrame(
        {
            "participant": df_estim["participant"],
            "rt_valid": rt_e.notna(),
            "is_slowfast": rt_e.notna() & ((rt_e < rt_fast) | (rt_e > rt_slow)),
        }
    )
    sf_e = (
        df_estim_sf.groupby("participant", observed=True)
        .agg(n_estim_trials=("rt_valid", "sum"), n_estim_slowfast=("is_slowfast", "sum"))
        .reset_index()
    )
    sf_e["frac_slowfast_estim"] = sf_e["n_estim_slowfast"] / sf_e["n_estim_trials"].replace(0, np.nan)

    sf = sf_c.merge(sf_e, on="participant", how="outer")
    sf["frac_bad_categ"] = sf["frac_bad_categ"].fillna(0.0)
    sf["frac_slowfast_estim"] = sf["frac_slowfast_estim"].fillna(0.0)
    sf["fail_slowfast_anytask"] = (sf["frac_bad_categ"] >= config.slowfast_frac_thresh) | (
        sf["frac_slowfast_estim"] >= config.slowfast_frac_thresh
    )
    sf = sf[
        ["participant", "fail_slowfast_anytask", "frac_bad_categ", "frac_slowfast_estim"]
    ].copy()

    rt_c_raw = pd.to_numeric(df_categ["rt"], errors="coerce")
    categ_rt_ok = rt_c_raw.notna() & (rt_c_raw >= rt_fast) & (rt_c_raw <= rt_slow)
    if config.use_trial_rt_exclusion:
        categ_keep = categ_rt_ok
    else:
        categ_keep = pd.Series(True, index=df_categ.index)
    if config.use_trial_valid_choice_exclusion:
        ch_raw = pd.to_numeric(df_categ["choice"], errors="coerce")
        categ_keep = categ_keep & ch_raw.isin([-1.0, 1.0])
    df_categ_clean = df_categ.loc[categ_keep].copy()

    if config.use_trial_rt_exclusion:
        rt_e_raw = pd.to_numeric(df_estim["rt"], errors="coerce")
        estim_keep = rt_e_raw.notna() & (rt_e_raw >= rt_fast) & (rt_e_raw <= rt_slow)
        df_estim_clean = df_estim.loc[estim_keep].copy()
    else:
        df_estim_clean = df_estim.copy()

    # --------------------------- CATEGORIZATION QC --------------------------- #
    df_categ_qc = df_categ_clean.copy()
    df_categ_qc["optimal_choice"] = (
        np.sign(df_categ_qc["sumllr_noisy"]) == np.sign(df_categ_qc["choice"])
    ).astype(int)

    cat_rows = []
    for pid, sub in df_categ_qc.groupby("participant"):
        n = len(sub)
        if n == 0:
            continue

        acc = float(sub["correct"].mean())
        acc_optim = float(
            (np.sign(sub["sumllr_noisy"]) == np.sign(sub["true_cat"]))
            .astype(int)
            .mean()
        )
        optim_choice = float(sub["optimal_choice"].mean())

        p_plus = float((sub["choice"] == 1).mean())
        same_dir_flag = max(p_plus, 1 - p_plus) >= config.same_dir_thresh

        med_rt = float(np.nanmedian(sub["rt"]))
        sd_rt = float(np.nanstd(sub["rt"], ddof=0))

        x = sub["sumllr_noisy"].to_numpy(float)
        y = (sub["choice"] == 1).astype(int).to_numpy()
        mu_hat, sigma_hat = fit_probit_glm(x, y)

        cat_rows.append(
            {
                "participant": pid,
                "n_trials_cat": n,
                "accuracy": acc,
                "acc_optim": acc_optim,
                "optim_choice": optim_choice,
                "prop_choice_plus1": p_plus,
                "same_dir_flag": same_dir_flag,
                "med_rt_choice": med_rt,
                "sd_rt_choice": sd_rt,
                "bias_mu": mu_hat,
                "sigma_hat": sigma_hat,
            }
        )

    res_cat = pd.DataFrame(cat_rows)
    res_cat = res_cat.merge(sf, on="participant", how="left")
    res_cat["fail_slowfast_anytask"] = res_cat["fail_slowfast_anytask"].fillna(False)


    # --------- Categorization bias/noise outliers (per incentive, MAD) --------- #
    if config.use_bias_mad_outlier_exclusion or config.use_noise_mad_outlier_exclusion:
        by_inc_rows = []
        for pid, sub_p in df_categ_qc.groupby("participant"):
            for inc in config.inc_levels:
                sub = sub_p[sub_p["incentive"] == inc]
                if len(sub) == 0:
                    by_inc_rows.append(
                        {
                            "participant": pid,
                            "incentive": inc,
                            "mu_inc": np.nan,
                            "sigma_inc": np.nan,
                        }
                    )
                    continue
                x = sub["sumllr_noisy"].to_numpy(float)
                y = (sub["choice"] == 1).astype(int).to_numpy()
                mu_i, sig_i = fit_probit_glm(x, y)
                by_inc_rows.append(
                    {"participant": pid, "incentive": inc, "mu_inc": mu_i, "sigma_inc": sig_i}
                )

        by_inc = pd.DataFrame(by_inc_rows)

        mu_w = by_inc.pivot(
            index="participant", columns="incentive", values="mu_inc"
        ).reindex(columns=config.inc_levels)
        mu_w.columns = [f"mu_{config.inc_names[c]}" for c in mu_w.columns]
        mu_w = mu_w.reset_index()

        sig_w = by_inc.pivot(
            index="participant", columns="incentive", values="sigma_inc"
        ).reindex(columns=config.inc_levels)
        sig_w.columns = [f"sigma_{config.inc_names[c]}" for c in sig_w.columns]
        sig_w = sig_w.reset_index()

        res_cat = res_cat.merge(mu_w, on="participant", how="left").merge(
            sig_w, on="participant", how="left"
        )

        for inc in config.inc_levels:
            if config.use_bias_mad_outlier_exclusion:
                res_cat[f"z_mu_{config.inc_names[inc]}"] = robust_z_series(
                    res_cat[f"mu_{config.inc_names[inc]}"]
                )
            if config.use_noise_mad_outlier_exclusion:
                s = res_cat[f"sigma_{config.inc_names[inc]}"].astype(float)
                if config.use_log_noise_for_mad:
                    s = np.log(s)
                res_cat[f"z_sigma_{config.inc_names[inc]}"] = robust_z_series(s)

        if config.use_bias_mad_outlier_exclusion:
            zcols = [f"z_mu_{config.inc_names[c]}" for c in config.inc_levels]
            res_cat["fail_bias_any_inc"] = (
                res_cat[zcols].abs().gt(config.z_mad).fillna(False).any(axis=1)
            )
        else:
            res_cat["fail_bias_any_inc"] = False

        if config.use_noise_mad_outlier_exclusion:
            zcols = [f"z_sigma_{config.inc_names[c]}" for c in config.inc_levels]
            res_cat["fail_noise_any_inc"] = (
                res_cat[zcols].abs().gt(config.z_mad).fillna(False).any(axis=1)
            )
        else:
            res_cat["fail_noise_any_inc"] = False
    else:
        res_cat["fail_bias_any_inc"] = False
        res_cat["fail_noise_any_inc"] = False

    if config.use_rt_sd_low_exclusion:
        res_cat["fail_rt_sd_low"] = res_cat["sd_rt_choice"] < config.rt_sd_min_s
    else:
        res_cat["fail_rt_sd_low"] = False

    if config.use_same_choice_repetition_exclusion:
        res_cat["fail_same_dir"] = res_cat["same_dir_flag"]
    else:
        res_cat["fail_same_dir"] = False

    from scipy.stats import binomtest

    if config.use_acc_above_optimal:
        # 3a) Above-optimal accuracy (binomial test), for at least one incentive
        above_opt_rows = []
        for (pid, inc), sub in df_categ_qc.groupby(["participant", "incentive"], observed=True):
            ev = pd.to_numeric(sub["sumllr_noisy"], errors="coerce").to_numpy(float)
            tc = pd.to_numeric(sub["true_cat"], errors="coerce").to_numpy(float)
            corr = pd.to_numeric(sub["correct"], errors="coerce").to_numpy(float)
            valid = np.isfinite(ev) & np.isfinite(tc) & np.isfinite(corr)
            n = int(valid.sum())
            if n <= 0:
                continue
            k = int(np.round(corr[valid].sum()))
            p_opt = float((np.sign(ev[valid]) == np.sign(tc[valid])).astype(int).mean())
            p_opt = float(np.clip(p_opt, 1e-6, 1.0 - 1e-6))
            pval = float(binomtest(k, n, p=p_opt, alternative="greater").pvalue)
            above_opt_rows.append(
                {"participant": pid, "incentive": inc, "pval_above_opt_vs_p_opt": pval}
            )

        above_opt_df = pd.DataFrame(above_opt_rows)
        if len(above_opt_df):
            fail_above_opt = (
                above_opt_df.groupby("participant", observed=True)["pval_above_opt_vs_p_opt"]
                .apply(lambda s: (s < config.alpha_cat).any())
                .reset_index(name="fail_above_opt_any_inc_binom")
            )
        else:
            fail_above_opt = pd.DataFrame(columns=["participant", "fail_above_opt_any_inc_binom"])
    else:
        fail_above_opt = pd.DataFrame(columns=["participant", "fail_above_opt_any_inc_binom"])

    # 3b) Not-above-chance optimality on easiest trials (sign(choice)==sign(evidence);
    # pooled across incentives), same easy subset as before.
    if config.use_easy_top_not_above_chance_exclusion:
        easy_fail_rows = []
        for pid, sub in df_categ_qc.groupby("participant", observed=True):
            ev = pd.to_numeric(sub["sumllr_noisy"], errors="coerce").to_numpy(float)
            ch = pd.to_numeric(sub["choice"], errors="coerce").to_numpy(float)
            valid = np.isfinite(ev) & np.isfinite(ch)
            ev = ev[valid]
            ch = ch[valid]
            optim = (np.sign(ev) == np.sign(ch)).astype(np.float64)
            n_total = int(len(ev))
            if n_total <= 0:
                easy_fail_rows.append(
                    {"participant": pid, "fail_easy_top_not_above_chance": True}
                )
                continue

            abs_ev = np.abs(ev)
            q = float(np.quantile(abs_ev, config.easy_quantile))
            easy_mask = abs_ev >= q
            n_easy = int(easy_mask.sum())
            min_easy = int(np.ceil(config.easy_top_frac * n_total))

            if n_easy <= 0 or n_easy < min_easy:
                fail_easy = False
            else:
                k_easy = int(np.round(optim[easy_mask].sum()))
                pval = float(binomtest(k_easy, n_easy, p=0.5, alternative="greater").pvalue)
                fail_easy = pval >= config.alpha_cat

            easy_fail_rows.append(
                {"participant": pid, "fail_easy_top_not_above_chance": bool(fail_easy)}
            )

        fail_easy_df = pd.DataFrame(easy_fail_rows)
    else:
        fail_easy_df = pd.DataFrame(columns=["participant", "fail_easy_top_not_above_chance"])

    # 3c) Psychometric slope <= 0 (sigma_hat from overall probit fit)
    res_cat["fail_slope_leq_0"] = (~np.isfinite(res_cat["sigma_hat"])) | (res_cat["sigma_hat"] <= 0)

    # Combine
    res_cat = res_cat.merge(fail_above_opt, on="participant", how="left")
    res_cat = res_cat.merge(fail_easy_df, on="participant", how="left")

    res_cat["fail_above_opt_any_inc_binom"] = res_cat["fail_above_opt_any_inc_binom"].fillna(False)
    res_cat["fail_easy_top_not_above_chance"] = res_cat["fail_easy_top_not_above_chance"].fillna(False)

    exclude_mask = pd.Series(False, index=res_cat.index)

    # Slow/fast criterion
    if config.use_slowfast_exclusion:
        exclude_mask = exclude_mask | res_cat["fail_slowfast_anytask"].fillna(False)

    # New participant-level exclusion criteria
    if config.use_acc_above_optimal:
        exclude_mask = exclude_mask | res_cat["fail_above_opt_any_inc_binom"].fillna(False)
    if config.use_easy_top_not_above_chance_exclusion:
        exclude_mask = exclude_mask | res_cat["fail_easy_top_not_above_chance"].fillna(False)
    if config.use_slope_leq_0_exclusion:
        exclude_mask = exclude_mask | res_cat["fail_slope_leq_0"].fillna(False)

    if config.use_same_choice_repetition_exclusion:
        exclude_mask = exclude_mask | res_cat["fail_same_dir"].fillna(False)
    if config.use_rt_sd_low_exclusion:
        exclude_mask = exclude_mask | res_cat["fail_rt_sd_low"].fillna(False)
    if config.use_bias_mad_outlier_exclusion:
        exclude_mask = exclude_mask | res_cat["fail_bias_any_inc"].fillna(False)
    if config.use_noise_mad_outlier_exclusion:
        exclude_mask = exclude_mask | res_cat["fail_noise_any_inc"].fillna(False)

    res_cat["exclude_cat"] = exclude_mask

    res_cat.to_csv(
        f"{config.outdir}/participant_qc_categ_simplified.csv", index=False
    )

    # ------------------------------ ESTIMATION QC ------------------------------ #
    df_estim_qc = df_estim_clean.copy()
    df_estim_qc["resp_deg"] = np.rad2deg(df_estim_qc["resp_estim"].to_numpy(float))
    df_estim_qc["ang_noisy"] = np.rad2deg(df_estim_qc["estimang_noisy"].to_numpy(float))

    df_estim_qc["err_deg"] = wrap_err_deg(
        df_estim_qc["resp_deg"].to_numpy(float),
        df_estim_qc["ang_noisy"].to_numpy(float),
    )
    df_estim_qc["abs_err_deg"] = np.abs(df_estim_qc["err_deg"])

    mae_cell = (
        df_estim_qc.groupby(["participant", "incentive"])["abs_err_deg"]
        .mean()
        .reset_index(name="mae_deg")
    )

    mae_cell["z_mae_inc"] = np.nan
    for inc, subc in mae_cell.groupby("incentive"):
        mae_cell.loc[subc.index, "z_mae_inc"] = robust_z_series(subc["mae_deg"]).values

    mae_any = mae_cell.groupby("participant")["z_mae_inc"].apply(
        lambda s: np.nanmax(np.abs(s.values)) if len(s) else np.nan
    )
    mae_any = mae_any.rename("max_abs_z_mae_inc").reset_index()

    est_rows = []
    for pid, sub in df_estim_qc.groupby("participant"):
        med_rt = float(np.nanmedian(sub["rt"])) if len(sub) else np.nan
        sd_rt = float(np.nanstd(sub["rt"], ddof=0)) if len(sub) else np.nan
        mae_all = float(np.nanmean(sub["abs_err_deg"])) if len(sub) else np.nan
        est_rows.append(
            {"participant": pid, "med_rt_estim": med_rt, "sd_rt_estim": sd_rt, "mae_all": mae_all}
        )

    res_est = pd.DataFrame(est_rows).merge(mae_any, on="participant", how="left")
    res_est = res_est.merge(sf, on="participant", how="left")
    res_est["fail_slowfast_anytask"] = res_est["fail_slowfast_anytask"].fillna(False)

    res_est["fail_rt_sd_low"] = res_est["sd_rt_estim"] < config.rt_sd_min_s
    # Estimation-specific criteria:
    # 1) SD(dev_noisy) > threshold (deg)
    # 2) SD(response) < threshold (deg)
    est_sd_rows = []
    for pid, sub in df_estim_qc.groupby("participant", observed=True):
        err_deg = pd.to_numeric(sub["err_deg"], errors="coerce").to_numpy(float)
        err_deg = err_deg[np.isfinite(err_deg)]
        resp_deg = pd.to_numeric(sub["resp_deg"], errors="coerce").to_numpy(float)
        resp_deg = resp_deg[np.isfinite(resp_deg)]

        sd_dev_noisy_deg = float(np.nanstd(err_deg, ddof=0)) if len(err_deg) else np.nan
        sd_resp_deg = float(np.nanstd(resp_deg, ddof=0)) if len(resp_deg) else np.nan

        fail_sd_dev_noisy = (not np.isfinite(sd_dev_noisy_deg)) or (
            sd_dev_noisy_deg > float(config.estim_sd_dev_noisy_max_deg)
        )
        fail_sd_resp_low = (not np.isfinite(sd_resp_deg)) or (
            sd_resp_deg < float(config.estim_sd_response_min_deg)
        )
        est_sd_rows.append(
            {
                "participant": pid,
                "sd_dev_noisy_deg": sd_dev_noisy_deg,
                "sd_resp_deg": sd_resp_deg,
                "fail_sd_dev_noisy_gt_thresh": bool(fail_sd_dev_noisy),
                "fail_sd_resp_low": bool(fail_sd_resp_low),
            }
        )
    est_sd_df = pd.DataFrame(est_sd_rows)
    res_est = res_est.merge(est_sd_df, on="participant", how="left")
    res_est["fail_sd_dev_noisy_gt_thresh"] = res_est["fail_sd_dev_noisy_gt_thresh"].fillna(False)
    res_est["fail_sd_resp_low"] = res_est["fail_sd_resp_low"].fillna(False)

    # Estimation bias/noise MAD outliers using signed noisy error
    err_stats_rows = []
    for pid, sub in df_estim_qc.groupby("participant", observed=True):
        err_deg = pd.to_numeric(sub["err_deg"], errors="coerce").to_numpy(float)
        err_deg = err_deg[np.isfinite(err_deg)]
        bias_estim = float(np.nanmean(err_deg)) if len(err_deg) else np.nan
        noise_estim = float(np.nanstd(err_deg, ddof=0)) if len(err_deg) else np.nan
        err_stats_rows.append(
            {
                "participant": pid,
                "bias_estim_deg": bias_estim,
                "noise_estim_deg": noise_estim,
            }
        )
    err_stats_df = pd.DataFrame(err_stats_rows)
    res_est = res_est.merge(err_stats_df, on="participant", how="left")
    res_est["z_bias_estim"] = robust_z_series(res_est["bias_estim_deg"])
    noise_for_mad = res_est["noise_estim_deg"].astype(float)
    if config.use_log_noise_for_mad:
        noise_for_mad = np.log(noise_for_mad)
    res_est["z_noise_estim"] = robust_z_series(noise_for_mad)
    res_est["fail_bias_estim"] = res_est["z_bias_estim"].abs().gt(config.z_mad).fillna(False)
    res_est["fail_noise_estim"] = res_est["z_noise_estim"].abs().gt(config.z_mad).fillna(False)

    exclude_est_mask = pd.Series(False, index=res_est.index)
    if config.use_slowfast_exclusion:
        exclude_est_mask = exclude_est_mask | res_est["fail_slowfast_anytask"].fillna(False)
    if config.use_rt_sd_low_exclusion:
        exclude_est_mask = exclude_est_mask | res_est["fail_rt_sd_low"].fillna(False)
    if config.use_bias_mad_outlier_exclusion:
        exclude_est_mask = exclude_est_mask | res_est["fail_bias_estim"].fillna(False)
    if config.use_noise_mad_outlier_exclusion:
        exclude_est_mask = exclude_est_mask | res_est["fail_noise_estim"].fillna(False)
    if config.use_estim_sd_dev_noisy_exclusion:
        exclude_est_mask = exclude_est_mask | res_est["fail_sd_dev_noisy_gt_thresh"].fillna(False)
    if config.use_estim_sd_response_low_exclusion:
        exclude_est_mask = exclude_est_mask | res_est["fail_sd_resp_low"].fillna(False)
    res_est["exclude_est"] = exclude_est_mask

    res_est.to_csv(
        f"{config.outdir}/participant_qc_estim_simplified.csv", index=False
    )

    # ---------------------- Joint participant-level QC ---------------------- #
    all_participants = pd.DataFrame(
        {"participant": pd.unique(pd.concat([df_categ_clean, df_estim_clean])["participant"])}
    )

    joint = (
        all_participants.merge(
            res_cat[["participant", "exclude_cat"]], on="participant", how="left"
        ).merge(res_est[["participant", "exclude_est"]], on="participant", how="left")
    )

    joint[["exclude_cat", "exclude_est"]] = joint[["exclude_cat", "exclude_est"]].fillna(
        False
    )
    joint["exclude_overall"] = joint[["exclude_cat", "exclude_est"]].any(axis=1)

    bad_cat = joint.loc[joint["exclude_cat"], ["participant"]].sort_values(
        "participant"
    )
    bad_est = joint.loc[joint["exclude_est"], ["participant"]].sort_values(
        "participant"
    )
    bad_all = joint.loc[joint["exclude_overall"], ["participant"]].sort_values(
        "participant"
    )


    data_clean_cat = df_categ_clean.loc[
        ~df_categ_clean["participant"].isin(bad_all["participant"])
    ].copy()
    data_clean_est = df_estim_clean.loc[
        ~df_estim_clean["participant"].isin(bad_all["participant"])
    ].copy()

    data_clean_cat.to_csv(out_cat_path, index=False)
    data_clean_est.to_csv(out_est_path, index=False)

    if config.make_plots:
        _make_categ_plots(df_categ_qc, res_cat, config)
        _make_estim_plots(df_estim_qc, res_est, config)
        _make_rt_distribution_grid(
            data_clean_cat[["participant", "rt"]].copy(),
            data_clean_est[["participant", "rt"]].copy(),
            config,
        )

    return {
        "df_categ_clean": df_categ_clean,
        "df_estim_clean": df_estim_clean,
        "res_cat": res_cat,
        "res_est": res_est,
        "mae_cell": mae_cell,
        "joint": joint,
        "bad_cat": bad_cat,
        "bad_est": bad_est,
        "bad_all": bad_all,
        "data_clean_cat": data_clean_cat,
        "data_clean_est": data_clean_est,
    }


# -------------------------- Plotting helpers -------------------------- #

def _fit_psychometric_by_cell(
    df_categ: pd.DataFrame,
    min_trials: int = 15,
) -> pd.DataFrame:
    """
    Fit probit  psychometric per (participant, incentive, seqlen)
    via `fit_probit_glm` (same model as participant-level QC).

    Returns DataFrame with participant, incentive, seqlen, bias, sigma,
    slope (=1/sigma), abs_bias, n_trials. Failed or too-small cells have NaN params.
    """
    rows = []
    for (pid, inc, sl), sub in df_categ.groupby(["participant", "incentive", "seqlen"]):
        if len(sub) < min_trials:
            rows.append({
                "participant": pid, "incentive": inc, "seqlen": sl,
                "bias": np.nan, "sigma": np.nan,
                "slope": np.nan, "abs_bias": np.nan, "n_trials": len(sub),
            })
            continue
        x = sub["sumllr_noisy"].to_numpy(float)
        y = (sub["choice"] == 1).astype(int).to_numpy()
        mu, sigma = fit_probit_glm(x, y)
        slope = 1.0 / sigma if (np.isfinite(sigma) and sigma > 0) else np.nan
        abs_bias = np.abs(mu) if np.isfinite(mu) else np.nan
        rows.append({
            "participant": pid, "incentive": inc, "seqlen": sl,
            "bias": mu, "sigma": sigma,
            "slope": slope, "abs_bias": abs_bias, "n_trials": len(sub),
        })
    return pd.DataFrame(rows)


def _make_categ_plots(df_categ: pd.DataFrame, res_cat: pd.DataFrame, config: QCConfig) -> None:
    """Replicates categorization figures from the original script."""
    qc_cat = res_cat.sort_values("accuracy").reset_index(drop=True)
    colors = np.where(qc_cat["exclude_cat"], "tab:red", "tab:blue")

    fig, axes = plt.subplots(5, 1, figsize=(16, 8), sharex=True)

    axes[0].bar(range(len(qc_cat)), qc_cat["accuracy"], color=colors)
    y = qc_cat["acc_optim"].to_numpy()
    y_ext = np.concatenate(([y[0]], y, [y[-1]]))
    x_ext = np.concatenate(([-0.5], np.arange(len(qc_cat)), [len(qc_cat) - 0.5]))
    axes[0].step(x_ext, y_ext, where="mid", color="k", linestyle="-")
    axes[0].axhline(0.5, color="k", linestyle="--")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Acc.")

    axes[1].bar(range(len(qc_cat)), qc_cat["optim_choice"], color=colors)
    axes[1].axhline(0.5, color="k", linestyle="--")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Frac. optimal")

    axes[2].bar(range(len(qc_cat)), qc_cat["bias_mu"], color=colors)
    axes[2].axhline(0, color="k", linestyle="-")
    axes[2].set_ylabel("Bias (μ)")

    axes[3].bar(range(len(qc_cat)), qc_cat["sigma_hat"], color=colors)
    axes[3].set_ylabel("Noise (σ)")
    axes[3].set_xticks(range(len(qc_cat)))
    axes[3].set_xticklabels(qc_cat["participant"], rotation=90, fontsize=8)

    axes[4].bar(range(len(qc_cat)), qc_cat["med_rt_choice"], color=colors)
    axes[4].set_ylabel("Median RT (s)")
    axes[4].set_xticks(range(len(qc_cat)))
    axes[4].set_xticklabels(qc_cat["participant"], rotation=90, fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{config.outdir}/categ_barplots_sorted.png", dpi=200)
    plt.close(fig)

    # Psychometric panels
    pids_sorted = qc_cat["participant"].tolist()
    n = len(pids_sorted)
    ncols = config.n_cols_panels
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 2.5, nrows * 2.2),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_2d(axes)

    gfit = np.linspace(-6, 6, 400)

    for i, pid in enumerate(pids_sorted):
        r, c = divmod(i, ncols)
        ax = axes[r, c]

        sub = df_categ[df_categ["participant"] == pid]
        x = sub["sumllr_noisy"].to_numpy(float)
        y = (sub["choice"] == 1).astype(int).to_numpy()

        sw = sliding_psychometric(
            x,
            y,
            x_min=-6,
            x_max=6,
            step=1,
            window=0.8,
            kernel="uniform",
            counts_min=10,
            return_ci=False,
        )
        sizes = np.clip(sw["n_eff"], 5, 50)
        ax.scatter(sw["grid"], sw["p"], s=sizes, alpha=0.8)

        row = qc_cat.loc[qc_cat["participant"] == pid].iloc[0]
        mu, sigma = float(row["bias_mu"]), float(row["sigma_hat"])
        pfit = norm.cdf(gfit, loc=mu, scale=max(sigma, 1e-6))
        ax.plot(gfit, pfit, lw=2)

        ax.axhline(0.5, ls="--", lw=0.6, alpha=0.6)
        ax.axvline(0, ls=":", lw=0.6, alpha=0.5)

        color = "red" if bool(row["exclude_cat"]) else "black"
        ax.set_title(f"{pid}\nacc={row['accuracy']:.2f}\nμ={mu:.2f}, σ={sigma:.2f}", fontsize=11, color=color)

        ax.set_xlim(-6, 6)
        ax.set_ylim(0, 1)

    for j in range(n, nrows * ncols):
        fig.delaxes(axes.flatten()[j])

    fig.suptitle("Categorization psychometrics: sliding window + probit fit", y=0.995)
    fig.text(0.5, 0.02, "sumLLR (evidence)", ha="center")
    fig.text(0.04, 0.5, "P(choice = +1)", va="center", rotation="vertical")
    plt.tight_layout(rect=[0.05, 0.04, 1, 0.97])
    plt.savefig(f"{config.outdir}/categ_psychometric_panels.png", dpi=200)
    plt.close(fig)

    # ---------- Probit psychometric per (incentive × seqlen) per participant ---------- #
    df_psy_cell = _fit_psychometric_by_cell(df_categ, min_trials=15)
    inc_colors = {-1: "red", 0: "gray", 1: "green"}
    seqlens = np.sort(df_psy_cell["seqlen"].dropna().unique())
    n_sl = len(seqlens)
    base_styles = ["-", "--", "-.", ":"]
    seqlen_styles = [base_styles[i % len(base_styles)] for i in range(n_sl)]
    seqlen_alpha = np.linspace(0.45, 1.0, n_sl).tolist() if n_sl else []
    seqlen_style_map = dict(zip(seqlens, seqlen_styles))
    seqlen_alpha_map = dict(zip(seqlens, seqlen_alpha))

    gfit_cell = np.linspace(-6, 6, 300)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 2.5, nrows * 2.2),
        sharex=True, sharey=True,
    )
    axes = np.atleast_2d(axes)
    for i, pid in enumerate(pids_sorted):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        sub_psy = df_psy_cell[df_psy_cell["participant"] == pid]
        for _, row in sub_psy.iterrows():
            mu = float(row["bias"])
            sig = float(row["sigma"])
            if not (np.isfinite(mu) and np.isfinite(sig)):
                continue
            inc = row["incentive"]
            sl = row["seqlen"]
            color = inc_colors.get(inc, "gray")
            ls = seqlen_style_map.get(sl, "-")
            alpha = seqlen_alpha_map.get(sl, 0.85)
            p_curve = norm.cdf(gfit_cell, loc=mu, scale=max(sig, 1e-6))
            ax.plot(gfit_cell, p_curve, color=color, linestyle=ls, alpha=alpha, lw=1.5)

        ax.axhline(0.5, ls="--", lw=0.6, alpha=0.6, color="k")
        ax.axvline(0, ls=":", lw=0.6, alpha=0.5, color="k")
        row_qc = qc_cat.loc[qc_cat["participant"] == pid].iloc[0]
        tit_color = "red" if bool(row_qc["exclude_cat"]) else "black"
        ax.set_title(f"{pid}", fontsize=10, color=tit_color)
        ax.set_xlim(-6, 6)
        ax.set_ylim(0, 1)
    for j in range(n, nrows * ncols):
        fig.delaxes(axes.flatten()[j])

    legend_handles: list[Line2D] = []
    legend_labels: list[str] = []
    for inc in sorted(inc_colors.keys()):
        for sl in seqlens:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=inc_colors[inc],
                    linestyle=seqlen_style_map.get(sl, "-"),
                    alpha=seqlen_alpha_map.get(sl, 0.85),
                    lw=1.5,
                )
            )
            legend_labels.append(f"inc={inc}, L={sl}")
    ncol = min(9, max(1, len(legend_handles)))
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=ncol,
        fontsize=7,
        frameon=True,
    )
    fig.suptitle(
        "Probit psychometric by incentive (color) and sequence length (linestyle, alpha)",
        y=0.995,
    )
    fig.text(0.5, 0.02, "sumLLR (evidence)", ha="center")
    fig.text(0.04, 0.5, "P(choice = +1)", va="center", rotation="vertical")
    plt.tight_layout(rect=[0.05, 0.06, 1, 0.97])
    plt.savefig(f"{config.outdir}/categ_psychometric_by_cell.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _make_estim_plots(df_estim: pd.DataFrame, res_est: pd.DataFrame, config: QCConfig) -> None:
    """Replicates estimation figures from the original script."""
    qc_est = res_est.sort_values("mae_all").reset_index(drop=True)
    colors = np.where(qc_est["exclude_est"], "tab:red", "tab:blue")

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].bar(range(len(qc_est)), qc_est["mae_all"], color=colors)
    axes[0].set_ylabel("Mean abs error (deg)")

    axes[1].bar(range(len(qc_est)), qc_est["med_rt_estim"], color=colors)
    axes[1].set_ylabel("Median RT (s)")
    axes[1].set_xticks(range(len(qc_est)))
    axes[1].set_xticklabels(qc_est["participant"], rotation=90, fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{config.outdir}/estim_barplots_sorted.png", dpi=200)
    plt.close(fig)

    # Error histogram panels
    pids_sorted = qc_est["participant"].tolist()
    n = len(pids_sorted)
    ncols = config.n_cols_panels
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 2.5, nrows * 2.0),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_2d(axes)

    bins = np.linspace(-90, 90, 19)  # 10-deg bins

    for i, pid in enumerate(pids_sorted):
        r, c = divmod(i, ncols)
        ax = axes[r, c]

        sub = df_estim[df_estim["participant"] == pid]
        err = sub["err_deg"].to_numpy(float)

        ax.hist(err, bins=bins, alpha=0.9)
        row = qc_est.loc[qc_est["participant"] == pid].iloc[0]
        color = "red" if bool(row["exclude_est"]) else "black"
        ax.set_title(f"{pid}\nMAE={row['mae_all']:.1f}", fontsize=11, color=color)
        ax.axvline(0, ls=":", lw=0.8, alpha=0.6)
        ax.set_xlim(-90, 90)

    for j in range(n, nrows * ncols):
        fig.delaxes(axes.flatten()[j])

    fig.suptitle("Estimation error histograms (signed error, wrapped to [-90, 90])", y=0.995)
    fig.text(0.5, 0.02, "Signed error (deg)", ha="center")
    fig.text(0.04, 0.5, "Count", va="center", rotation="vertical")
    plt.tight_layout(rect=[0.05, 0.04, 1, 0.97])
    plt.savefig(f"{config.outdir}/estim_error_hist_panels.png", dpi=200)
    plt.close(fig)


def _make_rt_distribution_grid(
    df_cat: pd.DataFrame,
    df_est: pd.DataFrame,
    config: QCConfig,
) -> None:
    """
    Grid of participant RT distributions after cleaning.

    df_cat and df_est must each have columns 'participant' and 'rt'.
    Produces two figures: one for categorization RT, one for estimation RT.
    """
    n_cols = config.n_cols_panels

    for task_name, df_task in [("categorization", df_cat), ("estimation", df_est)]:
        pids = df_task["participant"].unique()
        pids = np.sort(pids)
        n_p = len(pids)
        if n_p == 0:
            continue
        n_rows = int(np.ceil(n_p / n_cols))

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(2.2 * n_cols, 2.0 * n_rows),
            sharex=True,
            sharey=True,
        )
        axes = np.atleast_2d(axes)

        for idx, pid in enumerate(pids):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]
            rts = df_task.loc[df_task["participant"] == pid, "rt"].dropna()
            if len(rts) > 0:
                ax.hist(rts, bins=25, color="steelblue", edgecolor="white", alpha=0.9)
            ax.set_title(str(pid), fontsize=9)
            ax.set_ylabel("Count")

        for j in range(n_p, n_rows * n_cols):
            fig.delaxes(axes.flatten()[j])

        fig.suptitle(f"RT distributions after cleaning ({task_name})", y=1.01)
        for ax in axes.flat:
            if ax.get_figure() is not None:
                ax.set_xlabel("RT (s)")
        plt.tight_layout()
        plt.savefig(f"{config.outdir}/rt_distributions_after_clean_{task_name}.png", dpi=200)
        plt.close(fig)

