"""Preregistered behavioral analysis pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import circstd

from .stats import fit_slopes_intercepts, probit_sigma2


def _build_choice_variability_tables(df_inf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_inf = df_inf.copy()
    df_inf["optim"] = (np.sign(df_inf["choice"]) == np.sign(df_inf["sumllr_noisy"])).astype(int)

    df_var_inc = (
        df_inf.groupby(["participant", "incentive"])
        .apply(lambda d: probit_sigma2(d["sumllr_noisy"], d["choice"])[0])
        .reset_index(name="choice_var")
    )
    df_var_inc["sensitivity"] = 1.0 / np.sqrt(df_var_inc["choice_var"])

    df_var_inc_len = (
        df_inf.groupby(["participant", "incentive", "seqlen"])
        .apply(lambda d: probit_sigma2(d["sumllr_noisy"], d["choice"])[0])
        .reset_index(name="choice_var")
    )

    df_var_inc_len["choice_var_log"] = np.log(df_var_inc_len["choice_var"])
    return df_var_inc, df_var_inc_len


def _build_mean_tables(df_inf: pd.DataFrame, df_est: pd.DataFrame, df_var_inc: pd.DataFrame, df_var_inc_len: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_est = df_est.copy()
    df_m_ref = (
        df_est.groupby(["participant", "incentive"])["dev_noisy"]
        .agg(estim_sd=lambda x: circstd(x, high=np.pi, low=-np.pi))
        .reset_index()
    )
    df_m_ref["estim_sd"] = np.rad2deg(df_m_ref["estim_sd"])
    df_m_ref_rt = df_est.groupby(["participant", "incentive"])["rt"].agg(rt_estim_median="median").reset_index()
    df_m_ref = df_m_ref.merge(df_m_ref_rt, on=["participant", "incentive"])

    cols = ["participant", "incentive", "seqlen", "correct", "optim", "rt"]
    df_m = df_inf[cols].groupby(["participant", "incentive"]).mean().reset_index()
    df_m = df_m.merge(
        df_inf[["participant", "incentive", "rt"]].groupby(["participant", "incentive"]).median().reset_index(),
        how="left",
        on=["participant", "incentive"],
        suffixes=("", "_median"),
    )
    df_m = df_m.merge(df_m_ref, how="left", on=["participant", "incentive"])
    df_m = df_m.merge(df_var_inc, how="left", on=["participant", "incentive"])
    df_m["choice_var_log"] = np.log(df_m["choice_var"])
    df_m["choice_sd"] = 1.0 / np.sqrt(df_m["choice_var"])


    df_m_len = df_inf[cols].groupby(["participant", "incentive", "seqlen"]).mean().reset_index()
    df_m_len = df_m_len.merge(
        df_inf[["participant", "incentive", "seqlen", "rt"]]
        .groupby(["participant", "incentive", "seqlen"])
        .median()
        .reset_index(),
        how="left",
        on=["participant", "incentive", "seqlen"],
        suffixes=("", "_median"),
    )
    df_m_len = df_m_len.merge(df_var_inc_len, how="left", on=["participant", "incentive", "seqlen"])
    df_m_len["choice_var_log"] = np.log(df_m_len["choice_var"])
    return df_m, df_m_len


def _paired_abs_ttest(df: pd.DataFrame, dv: str, alternative: str = "two-sided") -> pd.DataFrame:
    """Paired t-test on abs incentive (1 vs 0), participant-collapsed."""
    d = df[["participant", "incentive_abs", dv]].dropna()
    d = d.groupby(["participant", "incentive_abs"], as_index=False)[dv].mean()
    wide = d.pivot(index="participant", columns="incentive_abs", values=dv)
    if 0 not in wide.columns or 1 not in wide.columns:
        return pd.DataFrame(
            {
                "test": [f"{dv}: abs incentive 1 vs 0"],
                "error": ["missing abs incentive level 0 or 1"],
            }
        )
    t = pg.ttest(wide[1], wide[0], paired=True, alternative=alternative)
    t.insert(0, "test", f"{dv}: abs incentive 1 vs 0")
    return t


def _run_analyses(df_m: pd.DataFrame, df_m_len: pd.DataFrame) -> dict[str, pd.DataFrame]:
    df_m = df_m.copy()
    df_m_len = df_m_len.copy()
    df_m["incentive_abs"] = df_m["incentive"].abs()
    df_m_len["incentive_abs"] = df_m_len["incentive"].abs()

    analyses: dict[str, pd.DataFrame] = {}

    # rmANOVA of sensitivity on incentives (+post-hoc)
    a_sens = pg.rm_anova(data=df_m, dv="sensitivity", within=["incentive"], subject="participant", detailed=True, effsize="np2")
    a_sens.post_hoc = pg.pairwise_tests(dv="sensitivity", within=["incentive"], subject="participant", data=df_m, padjust="holm", effsize="cohen")
    analyses["rm_anova_sensitivity_incentive"] = a_sens

    # one-way t-test of sensitivity on abs incentive (greater for 1 vs 0)
    analyses["ttest_sensitivity_abs_1_gt_0"] = _paired_abs_ttest(df_m, dv="sensitivity", alternative="greater")


    # rmANOVA of log(choice_var) on seqlen x incentive (+post-hoc pairwise incentive, Holm)
    a_cv = pg.rm_anova(data=df_m_len, dv="choice_var_log", within=["seqlen", "incentive"], subject="participant", detailed=True, effsize="np2")
    analyses["rm_anova_choice_var_log_seqlen_x_incentive"] = a_cv
    # rmANOVA of sd_estim on incentives (+post-hoc)
    a_estim_sd = pg.rm_anova(data=df_m, dv="estim_sd", within=["incentive"], subject="participant", detailed=True, effsize="np2")
    a_estim_sd.post_hoc = pg.pairwise_tests(dv="estim_sd", within=["incentive"], subject="participant", data=df_m, padjust="holm", effsize="cohen")
    analyses["rm_anova_estim_sd_incentive"] = a_estim_sd

    # paired t-test on estim_sd: abs incentive 1 vs 0 (alternative='less')
    analyses["ttest_estim_sd_abs_1_vs_0"] = _paired_abs_ttest(df_m, dv="estim_sd", alternative="less")
    return analyses


def _df_to_simple_md(df: pd.DataFrame) -> str:
    rounded = df.round(3)
    try:
        return rounded.to_markdown(index=True, tablefmt="github")
    except ImportError:
        return "```\n" + rounded.to_string(index=True) + "\n```"


def run_behavior_main_prereg(exp_name: str = "exp1") -> None:
    base = Path("data/processed") / exp_name
    df_inf = pd.read_csv(base / "data_clean_inf.csv")
    df_est = pd.read_csv(base / "data_clean_est.csv")

    df_var_inc, df_var_inc_len = _build_choice_variability_tables(df_inf)
    df_m, df_m_len = _build_mean_tables(df_inf, df_est, df_var_inc, df_var_inc_len)
    # Kept for compatibility/debug table output only.
    slopes_df = fit_slopes_intercepts(df_var_inc_len, dv="choice_var", iv="seqlen", verbose=True)
    df_var_slopes = df_var_inc.merge(slopes_df, on=["participant", "incentive"], how="left")

    results_dir = Path("results/prereg") / exp_name / "behavior"
    results_dir.mkdir(parents=True, exist_ok=True)
    df_m.to_csv(results_dir / "data_mean_inc.csv", index=False)
    df_m_len.to_csv(results_dir / "data_mean_inc_seqlen.csv", index=False)
    df_var_slopes.to_csv(results_dir / "behav_var_slopes.csv", index=False)

    analyses = _run_analyses(df_m, df_m_len)
    md_parts = ["# Behavioral statistics (prereg)\n"]
    for key, tbl in analyses.items():
        md_parts.append(f"## {key}\n")
        md_parts.append(_df_to_simple_md(tbl))
        md_parts.append("")
        post = getattr(tbl, "post_hoc", None)
        if post is not None:
            md_parts.append("### post-hoc\n")
            md_parts.append(_df_to_simple_md(post))
            md_parts.append("")
        post_abs = getattr(tbl, "post_hoc_abs", None)
        if post_abs is not None:
            md_parts.append("### post-hoc (abs)\n")
            md_parts.append(_df_to_simple_md(post_abs))
            md_parts.append("")
    (results_dir / "behav_stats.md").write_text("\n".join(md_parts).rstrip() + "\n", encoding="utf-8")
