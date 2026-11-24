"""Variance partition of estimation errors and inference noise into bias and noise (Figure 4, Tables 2-3).

Estimation: per participant x incentive, test-retest split of circular errors to repeated stimuli
(``compute_variance_partition``): bias share ``rho`` (= the test-retest correlation ``rho_testretest``
clipped to [0.001, 0.999]) and bias / noise / total SDs.

Integration: the fitted single-noise sigma_inf (prereg ``fit_noise_per_cond.npz``) is split with the
repeated-sequence likelihood of Drugowitsch et al. (2016): per participant x incentive, the bias share
``rho`` (= correlation between presentations of the same sequence) is fitted from the joint choices to
repeated sequences (Beta(1.1, 1.1) MAP, clipped to [0.001, 0.999]; ``rho_mle`` and a profile-likelihood
95% interval alongside), and ``bias_var = rho x sigma_inf^2``. ``consistency`` is the share of repeated
sequences answered like their first presentation.

Bias / noise / total SDs are floored at 1e-4 (``split_total_variance``), so no stored component is exactly
0; the ``*_sd_log`` columns are their natural logs.

Outputs: ``results/exploratory/<exp>/variance_partition/variance_partition_estim.csv`` and
``var_partition_categ.csv`` (read by ``paper_tables.py`` and ``scripts/exploratory/figures/fig4.py``).

Usage (from repo root), once per experiment:

    python scripts/exploratory/stats/run_variance_partition.py --exp-name exp1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

from motinf.exploratory.stats import compute_consistency
from motinf.exploratory.variance_partition import (
    compute_variance_partition,
    fit_bias_share_repeats,
    split_total_variance,
)


def _log_sd_series(sd: pd.Series) -> pd.Series:
    """Natural log of an SD column (the stored SDs are floored at ``SD_FLOOR`` > 0 by ``split_total_variance``)."""
    return pd.Series(np.log(pd.to_numeric(sd, errors="coerce").to_numpy(dtype=float)), index=sd.index)


def _prepare_inference_repeat_columns(df_inf: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    """Add a sequence identity (hash of the noisy sample angles) and a first-vs-repeat label (0 for the
    first presentation in trial order, 1 for later ones); return the table and both column names."""
    dfi = df_inf.copy()
    seq_cols = [c for c in dfi.columns if c.startswith("seqang_noisy_")]
    seq_sig = dfi[seq_cols].apply(pd.to_numeric, errors="coerce").round(8)
    dfi["_stim_key_derived"] = pd.util.hash_pandas_object(seq_sig, index=False).astype(str)
    stim_col = "_stim_key_derived"

    dfi["_orig_order"] = np.arange(len(dfi))
    dfi = dfi.sort_values(["participant", "incentive", "session", "date", "_orig_order"])
    dfi["_block_rep_derived"] = (
        dfi.groupby(["participant", "incentive", stim_col], observed=True)
        .cumcount()
        .clip(upper=1)
    )
    dfi = dfi.sort_values("_orig_order").drop(columns=["_orig_order"])
    return dfi, stim_col, "_block_rep_derived"


def _load_fitted_inference_variance(exp_name: str) -> pd.DataFrame:
    """
    Load fitted inference noise from prereg model fits and return
    participant × incentive fitted sd_inf plus base variance proxy (sd_inf^2).
    """
    fit_path = ROOT / "results" / "prereg" / exp_name / "model_fit" / "fit_noise_per_cond.npz"
    if not fit_path.exists():
        raise FileNotFoundError(f"Fitted inference noise file not found: {fit_path}")

    fit = np.load(fit_path, allow_pickle=True)
    required = {"pars_inf", "participants", "incentives"}
    missing = required.difference(set(fit.files))
    if missing:
        raise ValueError(
            f"{fit_path} is missing required arrays: {sorted(missing)}."
        )

    participants = pd.Series(fit["participants"], dtype=object).astype(str).to_numpy()
    incentives = np.asarray(fit["incentives"], dtype=float)
    pars_inf = np.asarray(fit["pars_inf"], dtype=object)

    if pars_inf.ndim != 2:
        raise ValueError(f"Expected pars_inf to be 2D, got shape {pars_inf.shape}.")
    if pars_inf.shape[0] != participants.shape[0]:
        raise ValueError(
            "Mismatch between participants and pars_inf rows: "
            f"{participants.shape[0]} vs {pars_inf.shape[0]}."
        )
    if pars_inf.shape[1] != incentives.shape[0]:
        raise ValueError(
            "Mismatch between incentives and pars_inf columns: "
            f"{incentives.shape[0]} vs {pars_inf.shape[1]}."
        )

    # npz stores each cell as a length-1 array; extract scalar safely.
    sd_inf = np.vectorize(
        lambda x: float(np.asarray(x, dtype=float).reshape(-1)[0]),
        otypes=[float],
    )(pars_inf)
    if not np.all(np.isfinite(sd_inf)):
        raise ValueError("pars_inf contains non-finite values.")

    df_sd = pd.DataFrame(sd_inf, columns=incentives)
    df_sd.insert(0, "participant", participants)
    df_sd = df_sd.dropna(subset=["participant"]).copy()

    df_sd_long = df_sd.melt(
        id_vars="participant",
        var_name="incentive",
        value_name="sd_inf_fit",
    )
    df_sd_long["incentive"] = pd.to_numeric(df_sd_long["incentive"], errors="coerce")
    df_sd_long["choice_var_sdinf"] = df_sd_long["sd_inf_fit"] ** 2
    return df_sd_long[["participant", "incentive", "sd_inf_fit", "choice_var_sdinf"]]


def standardized_evidence(
    df_inf_rep: pd.DataFrame,
    df_var_fit: pd.DataFrame,
    *,
    seqlen_m1_floor: float = 1.0,
) -> pd.DataFrame:
    """
    Inference trials with fitted ``sd_inf_fit`` merged in and the standardized evidence
    ``h_std = sumllr_noisy / (sd_inf * sqrt(max(seqlen - 1, floor)))`` (the input of the bias-share fit).
    """
    dfi = df_inf_rep.dropna(subset=["choice", "sumllr_noisy", "seqlen"]).copy()
    dfi["participant"] = dfi["participant"].astype(str)
    dfi["incentive"] = pd.to_numeric(dfi["incentive"], errors="coerce")
    fit = df_var_fit.assign(participant=lambda d: d["participant"].astype(str))
    dfi = dfi.merge(fit[["participant", "incentive", "sd_inf_fit"]], on=["participant", "incentive"], how="inner")
    m = np.maximum(pd.to_numeric(dfi["seqlen"]).to_numpy(float) - 1.0, float(seqlen_m1_floor))
    dfi["h_std"] = dfi["sumllr_noisy"].to_numpy(float) / (dfi["sd_inf_fit"].to_numpy(float) * np.sqrt(m))
    return dfi


CATEG_CSV_COLS = [
    "participant",
    "incentive",
    "choice_var",
    "consistency",
    "rho",
    "bias_var",
    "noise_var",
    "noise_sd",
    "bias_sd",
    "total_sd",
    "noise_sd_log",
    "bias_sd_log",
    "rho_mle",
    "rho_ci_lo",
    "rho_ci_hi",
    "n_repeat_pairs",
    "n_repeat_multi",
]


def likelihood_inference_split(
    df_inf_rep: pd.DataFrame,
    df_var_fit: pd.DataFrame,
    stim_col: str,
    *,
    seqlen_m1_floor: float = 1.0,
    prior: tuple[float, float] = (1.1, 1.1),
) -> pd.DataFrame:
    """
    Primary categorization split. Per participant x incentive, the bias share ``rho`` of the
    fitted inference noise is estimated from the choices to repeated sequences with the
    repeated-sequence likelihood of Drugowitsch et al. (2016)
    (:func:`motinf.exploratory.variance_partition.fit_bias_share_repeats`), reported as the
    posterior mode under a Beta(*prior) prior, with the MLE and its profile-likelihood 95% interval.

    Each trial's standardized evidence is ``sumllr_noisy / (sd_inf * sqrt(max(seqlen - 1, floor)))``,
    i.e. the mean and SD of the prereg inference model fitted per condition (sd_inf only; no
    lapse, bias or other noise terms). ``rho`` is both the bias share and the correlation
    between presentations, so ``bias_var = rho * sd_inf^2`` and ``noise_var = (1 - rho) * sd_inf^2``.

    """
    dfi = standardized_evidence(df_inf_rep, df_var_fit, seqlen_m1_floor=seqlen_m1_floor)

    rows: list[dict] = []

    for (pid, inc), sub in dfi.groupby(["participant", "incentive"], observed=True, sort=True):
        res = fit_bias_share_repeats(
            sub["h_std"].to_numpy(),
            sub["choice"].to_numpy() == 1,
            sub[stim_col].to_numpy(),
            prior=prior,
        )
        # bias share clipped to RHO_BOUNDS and SDs floored at SD_FLOOR when saving
        split = split_total_variance(res["rho_map"], float(sub["sd_inf_fit"].iloc[0]) ** 2)
        rows.append(
            {
                "participant": pid,
                "incentive": float(inc),
                "rho": float(split["rho"]),
                "rho_mle": res["rho_mle"],
                "rho_ci_lo": res["ci_lo"],
                "rho_ci_hi": res["ci_hi"],
                "n_repeat_pairs": res["n_pairs"],
                "n_repeat_multi": res["n_multi"],
                "bias_var": float(split["bias_var"]),
                "noise_var": float(split["noise_var"]),
            }
        )
    return pd.DataFrame(rows)


def run_for_experiment(exp_name: str) -> None:
    df_est = pd.read_csv(ROOT / "data" / "processed" / exp_name / "data_clean_est.csv")
    df_inf = pd.read_csv(ROOT / "data" / "processed" / exp_name / "data_clean_inf.csv")
    outdir = ROOT / "results" / "exploratory" / exp_name / "variance_partition"
    outdir.mkdir(parents=True, exist_ok=True)

    # Estimation: test-retest split over repeated stimuli
    df_corrsplit = compute_variance_partition(df_est, stim_col="gaborimg_estim")
    df_corrsplit["noise_sd_log"] = _log_sd_series(df_corrsplit["noise_sd"])
    df_corrsplit["bias_sd_log"] = _log_sd_series(df_corrsplit["bias_sd"])
    df_corrsplit.to_csv(outdir / "variance_partition_estim.csv", index=False)

    # Integration: repeated-sequence likelihood split of the fitted sigma_inf^2
    df_var_fit = _load_fitted_inference_variance(exp_name)
    df_inf_rep, inf_stim_col, inf_rep_col = _prepare_inference_repeat_columns(df_inf)
    df_inf_cons = compute_consistency(df_inf_rep, stim_col=inf_stim_col, rep_col=inf_rep_col)
    df_cons = (
        df_inf_cons.groupby(["participant", "incentive"])["consistency"]
        .mean()
        .reset_index(name="consistency")
    )
    df_cons["participant"] = df_cons["participant"].astype(str)
    df_cons["incentive"] = pd.to_numeric(df_cons["incentive"], errors="coerce")

    base = df_var_fit.assign(participant=lambda d: d["participant"].astype(str)).rename(
        columns={"choice_var_sdinf": "choice_var"}
    )
    lik_tbl = likelihood_inference_split(df_inf_rep, df_var_fit, inf_stim_col)
    res_inf = base.merge(df_cons, on=["participant", "incentive"], how="left").merge(
        lik_tbl, on=["participant", "incentive"], how="left"
    )

    # ``choice_var`` is sigma_inf^2, so total_sd = sigma_inf.
    res_inf["noise_sd"] = (res_inf["noise_var"].clip(lower=0.0)) ** 0.5
    res_inf["bias_sd"] = (res_inf["bias_var"].clip(lower=0.0)) ** 0.5
    res_inf["total_sd"] = (res_inf["choice_var"].clip(lower=0.0)) ** 0.5
    res_inf["noise_sd_log"] = _log_sd_series(res_inf["noise_sd"])
    res_inf["bias_sd_log"] = _log_sd_series(res_inf["bias_sd"])
    res_inf[CATEG_CSV_COLS].to_csv(outdir / "var_partition_categ.csv", index=False)
    print(f"Wrote {outdir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bias/noise variance partition (estimation and integration).")
    parser.add_argument("--exp-name", default="exp1", help="exp1 or exp2.")
    run_for_experiment(parser.parse_args().exp_name)


if __name__ == "__main__":
    main()
