"""Reaction-time control analyses (Figure 5, Supplementary Tables 8-10).

Writes tables to ``results/exploratory/<exp>/behavior/stats/rt_analysis/``; Figure 5 is drawn from
them by ``scripts/exploratory/figures/fig5.py`` and the statistics are tabulated by ``paper_tables.py``.

- Median RT per participant x incentive (Estimation and Integration) and its gain/loss - neutral
  contrasts (Figure 5A).
- Within-participant median split of trials into fast and slow: estimation error SD (circular) and
  sigma_inf (``fit_inf_rt_split.npz``, from ``fit_inf_per_cond_rt_split.py``) per participant x RT bin x
  incentive. Tested: the incentive effect fast vs slow (paired t, Holm over gain/loss; Supplementary
  Table 9) and the noise difference fast - slow within each incentive (one-sample t, Holm over the three
  levels; Figure 5C).
- Per-participant RT and noise contrasts paired for the RT-noise correlations (Supplementary Table 10).
- RT as a function of evidence (Figure 5B, Supplementary Table 8): see ``rt_regression.py``.

Per-participant incentive contrasts are ``value(incentive) - value(neutral)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import circstd

from motinf.exploratory.cohorts import (
    cohort_subexperiments,
    merge_behavior_mean_inc,
    merge_data_clean_est,
    merge_data_clean_inf,
    merge_sd_inf_long,
    merge_sd_inf_rt_split_long,
)
from motinf.exploratory.rt_regression import run_rt_evidence_regression


RT_BINS = ["fast", "slow"]
CONTRASTS = {
    "gain_minus_neutral": (1, 0),
    "loss_minus_neutral": (-1, 0),
}


def rt_stats_dir(root: Path, exp_name: str) -> Path:
    """Folder of the RT analysis tables (written by :func:`run_response_times`, read by Figure 5)."""
    return root / "results" / "exploratory" / exp_name / "behavior" / "stats" / "rt_analysis"


def _build_contrasts(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    keep = ["participant", "incentive", value_col]
    wide = (
        df[keep]
        .dropna()
        .drop_duplicates(subset=["participant", "incentive"])
        .pivot(index="participant", columns="incentive", values=value_col)
    )
    rows: list[dict[str, float | str]] = []
    for contrast_name, (a, b) in CONTRASTS.items():
        if a not in wide.columns or b not in wide.columns:
            continue
        eff = wide[a] - wide[b]
        for pid, val in eff.items():
            if np.isfinite(val):
                rows.append({"participant": pid, "contrast": contrast_name, "effect": float(val)})
    return pd.DataFrame(rows)


def _build_contrasts_by_bin(df: pd.DataFrame, value_col: str, bin_col: str = "rt_bin") -> pd.DataFrame:
    rows = []
    for rt_bin in RT_BINS:
        sub = df[df[bin_col] == rt_bin]
        c = _build_contrasts(sub, value_col=value_col)
        if c.empty:
            continue
        c.insert(1, bin_col, rt_bin)
        rows.append(c)
    if not rows:
        return pd.DataFrame(columns=["participant", bin_col, "contrast", "effect"])
    return pd.concat(rows, ignore_index=True)


def _build_sd_estim_split(df_est: pd.DataFrame) -> pd.DataFrame:
    df_est = df_est.copy()
    df_est["rt"] = pd.to_numeric(df_est["rt"], errors="coerce")
    df_est = df_est[df_est["rt"].notna() & (df_est["rt"] > 0)].copy()
    rt_med = df_est.groupby("participant")["rt"].transform("median")
    df_est["rt_bin"] = np.where(df_est["rt"] <= rt_med, "fast", "slow")
    agg = (
        df_est.groupby(["participant", "rt_bin", "incentive"])["dev_noisy"]
        .agg(sd_estim=lambda x: np.rad2deg(circstd(x, high=np.pi, low=-np.pi)))
        .reset_index()
    )
    agg["log_sd_estim"] = np.log(agg["sd_estim"])
    return agg


def _paired_contrasts(rt_contrasts: pd.DataFrame, noise_contrasts: pd.DataFrame, label: str) -> pd.DataFrame:
    """RT and noise contrasts of the same participant side by side (inputs to the RT-noise correlations)."""
    merged = rt_contrasts.merge(
        noise_contrasts,
        on=["participant", "contrast"],
        suffixes=("_rt", "_noise"),
        how="inner",
    )
    return merged.assign(correlation_label=label)


def _paired_t_fast_vs_slow(df_eff: pd.DataFrame, value_col: str, label: str) -> pd.DataFrame:
    out = []
    for contrast_name in CONTRASTS:
        d = df_eff[df_eff["contrast"] == contrast_name]
        wide = d.pivot(index="participant", columns="rt_bin", values=value_col)
        if "fast" not in wide.columns or "slow" not in wide.columns:
            out.append(pd.DataFrame({"metric": [label], "contrast": [contrast_name], "error": ["missing fast/slow columns"]}))
            continue
        wide = wide.dropna(subset=["fast", "slow"])
        if len(wide) < 2:
            out.append(pd.DataFrame({"metric": [label], "contrast": [contrast_name], "error": ["too few paired participants"]}))
            continue
        t = pg.ttest(wide["slow"], wide["fast"], paired=True, alternative="two-sided")
        t.insert(0, "metric", label)
        t.insert(1, "contrast", contrast_name)
        t.insert(2, "n", len(wide))
        out.append(t)
    return pd.concat(out, ignore_index=True)


def _build_fast_minus_slow_by_incentive(df_split: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Per-participant `fast - slow` difference of `value_col`, one row per raw incentive level
    (loss/neutral/gain), from the raw (pre-contrast) fast/slow split table.

    E.g. for a given participant: gain_fast - gain_slow, neutral_fast - neutral_slow,
    loss_fast - loss_slow. Distinct from :func:`_paired_t_fast_vs_slow`, which tests whether the
    *incentive effect* (already a gain/loss - neutral contrast) differs between fast and slow.
    """
    rows: list[dict[str, float]] = []
    for inc in sorted(df_split["incentive"].unique()):
        d = df_split[df_split["incentive"] == inc]
        wide = d.pivot(index="participant", columns="rt_bin", values=value_col)
        if "fast" not in wide.columns or "slow" not in wide.columns:
            continue
        diff = (wide["fast"] - wide["slow"]).dropna()
        for pid, val in diff.items():
            rows.append({"participant": pid, "incentive": float(inc), "diff": float(val)})
    return pd.DataFrame(rows, columns=["participant", "incentive", "diff"])


def _build_fast_minus_slow_incentive_effect(df_eff: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Per-participant `fast - slow` difference of the incentive effect on noise (`value_col`),
    one row per contrast (gain, loss).

    This is the effect size behind the :func:`_paired_t_fast_vs_slow` test (whose reported ``t``
    is on the opposite sign, ``slow - fast``, kept as-is there to match already-reported
    statistics); here the sign matches how the draft text describes it.
    """
    rows: list[dict[str, float | str]] = []
    for contrast_name in CONTRASTS:
        d = df_eff[df_eff["contrast"] == contrast_name]
        wide = d.pivot(index="participant", columns="rt_bin", values=value_col)
        if "fast" not in wide.columns or "slow" not in wide.columns:
            continue
        diff = (wide["fast"] - wide["slow"]).dropna()
        for pid, val in diff.items():
            rows.append({"participant": pid, "contrast": contrast_name, "diff": float(val)})
    return pd.DataFrame(rows, columns=["participant", "contrast", "diff"])


def _p_val_col(t: pd.DataFrame) -> str:
    """p-value column of a ``pg.ttest`` table (``p-val`` in pingouin 0.5, ``p_val`` in 0.6)."""
    return "p_val" if "p_val" in t.columns else "p-val"


def _ttest_fast_minus_slow_by_incentive(diff: pd.DataFrame, label: str) -> pd.DataFrame:
    """One-sample t-test of the per-participant ``fast - slow`` noise difference against 0 within each
    incentive level, Holm-corrected across the three levels (``p_holm``)."""
    rows = []
    for inc, g in diff.groupby("incentive"):
        x = g["diff"].dropna()
        t = pg.ttest(x, 0.0)
        rows.append({"metric": label, "incentive": float(inc), "n": len(x), "mean_fast_minus_slow": x.mean(),
                     "T": float(t["T"].iat[0]), "dof": float(t["dof"].iat[0]), "p-val": float(t[_p_val_col(t)].iat[0]),
                     "cohen-d": float(x.mean() / x.std(ddof=1))})
    out = pd.DataFrame(rows)
    out["p_holm"] = pg.multicomp(out["p-val"].to_numpy(), method="holm")[1]
    return out


def run_response_times(exp_name: str, *, root: Path) -> None:
    """Run the RT analyses for one experiment or the merged sample and write their tables."""
    cohort_segs = cohort_subexperiments(exp_name)
    stats_dir = rt_stats_dir(root, exp_name)
    stats_dir.mkdir(parents=True, exist_ok=True)

    df_mean = merge_behavior_mean_inc(root, cohort_segs)
    df_sd_inf = merge_sd_inf_long(root, cohort_segs)
    df_mean["log_sd_estim"] = np.log(df_mean["estim_sd"])
    df_mean[["participant", "incentive", "rt_median", "rt_estim_median"]].drop_duplicates().to_csv(
        stats_dir / "rt_median_by_incentive.csv", index=False
    )

    rt_estim_ctr = _build_contrasts(df_mean, "rt_estim_median")
    rt_estim_ctr.to_csv(stats_dir / "rt_effect_estim.csv", index=False)
    sd_estim_ctr = _build_contrasts(df_mean, "log_sd_estim").rename(columns={"effect": "effect_noise"})
    _paired_contrasts(rt_estim_ctr.rename(columns={"effect": "effect_rt"}), sd_estim_ctr, "rt_estim_vs_log_sd_estim").to_csv(
        stats_dir / "corr_input_rt_estim_vs_log_sd_estim.csv", index=False
    )

    rt_inf_ctr = _build_contrasts(df_mean, "rt_median")
    rt_inf_ctr.to_csv(stats_dir / "rt_effect_inf.csv", index=False)
    sd_inf_ctr = _build_contrasts(df_sd_inf, "sd_inf_log").rename(columns={"effect": "effect_noise"})
    _paired_contrasts(rt_inf_ctr.rename(columns={"effect": "effect_rt"}), sd_inf_ctr, "rt_median_vs_sd_inf_log").to_csv(
        stats_dir / "corr_input_rt_median_vs_sd_inf_log.csv", index=False
    )

    df_est = merge_data_clean_est(root, cohort_segs)
    df_sd_estim_split = _build_sd_estim_split(df_est)
    df_sd_inf_split = merge_sd_inf_rt_split_long(root, cohort_segs)
    for label, df_split, value_col in (
        ("log_sd_estim", df_sd_estim_split, "log_sd_estim"),
        ("sd_inf_log", df_sd_inf_split, "sd_inf_log"),
    ):
        split_eff = _build_contrasts_by_bin(df_split, value_col=value_col)
        split_eff.to_csv(stats_dir / f"split_effects_{label}.csv", index=False)
        t_split = _paired_t_fast_vs_slow(split_eff, value_col="effect", label=label)
        pcol = _p_val_col(t_split)  # Holm across the gain/loss contrasts
        ok = t_split[pcol].notna()
        t_split.loc[ok, "p_holm"] = pg.multicomp(t_split.loc[ok, pcol].to_numpy(), method="holm")[1]
        t_split.to_csv(stats_dir / f"ttest_split_{label}.csv", index=False)
        _build_fast_minus_slow_incentive_effect(split_eff, value_col="effect").to_csv(
            stats_dir / f"diff_fast_minus_slow_incentive_effect_{label}.csv", index=False
        )

        diff = _build_fast_minus_slow_by_incentive(df_split, value_col=value_col)
        diff.to_csv(stats_dir / f"diff_fast_minus_slow_{label}.csv", index=False)
        _ttest_fast_minus_slow_by_incentive(diff, label).to_csv(
            stats_dir / f"ttest_noise_fast_minus_slow_by_incentive_{label}.csv", index=False
        )

    run_rt_evidence_regression(merge_data_clean_inf(root, cohort_segs), stats_dir)
    print(f"Saved RT analysis tables to: {stats_dir}")
