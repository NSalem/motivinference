"""Statistics behind the paper's results tables (main Tables 1-3, several Supplementary Tables).

One procedure, three samples (``exp1``, ``exp2``, ``merged`` = pooled exp1+exp2), computed
directly from participant-level files rather than read back from per-table CSVs, so every
number in every table follows the same conventions:

- Contrasts are **incentive - neutral** (gain - neutral, loss - neutral, and the pooled
  mean(gain, loss) - neutral) and **gain - loss**, so a negative value on a noise measure always
  means "incentive reduces noise".
- Effect size is **dz** (mean paired difference / SD of paired differences = t / sqrt(n)) with an
  exact 95% CI from the noncentral t distribution.
- Pairwise p-values are Holm-corrected over the three contrasts gain-neutral, loss-neutral and
  gain-loss of a measure; the pooled contrast and the one-tailed |incentive| tests are uncorrected.
- Bayes factors are default JZS (Cauchy r = .707), two-sided, computed by pingouin
  (``bayesfactor_ttest``: Rouder et al., 2009, Eq. 1 / Note 4; ``bayesfactor_pearson`` for correlations).
  pingouin matched R's ``BayesFactor::meta.ttestBF`` within 4e-10 relative error at all 710 one-sample (t, n)
  pairs behind a reported BF (2026-11-19); a subset is pinned in ``tests/test_bayes_factor.py``. The one-tailed
  |incentive| tests of Table 1 carry no Bayes factor.
- Omnibus rmANOVA p-values are **always Greenhouse-Geisser corrected with the exact epsilon of each
  effect** (:mod:`motinf.prereg.rm_anova_exact`). pingouin's epsilon is wrong for interactions of
  two factors with > 2 levels, so pingouin's ``rm_anova`` is not used here.

Bias measure. Bias SD is tested on the **log scale** (``log_bias_sd``), like noise SD and Figure 4;
descriptives stay in native units. The stored partitions bound the bias share to [0.001, 0.999], so
no cell has zero bias (Estimation: ~5% of cells sit at the lower bound, whose log value depends on
the bound; Integration: none). The natural-scale bias SD tests are kept for the Table 3 note
(``supp.partition``: ``*_bias_natural``, pooled omnibus).

Table map (paper -> keys): Table 1 -> ``table1``; Table 2 -> ``table2`` (sensitivity, categ_optim,
log_sd_inf, estim_sd); Table 3 -> ``table2`` (categ_noise_log, categ_bias, estim_noise_log, estim_bias).
Supplement: Table 1 -> ``supp.choice_var_natural``, ``supp.estim_sd_log``; Table 2 -> ``table2`` (exp1,
exp2) + ``supp.heterogeneity`` (Experiment 1 vs 2 for every contrast); Table 3 and 5 ->
``supp.questionnaire``; Table 4 -> ``supp.choice_var_prediction``; Table 6 -> ``supp.biases``; Table 7
-> ``table2`` (categ_rt, estim_rt); Table 8 -> ``supp.rt_evidence``; Table 9 -> ``supp.fast_slow*``;
Table 10 -> ``supp.rt_corr``. Partition omnibus tests quoted in the Results (correlation between
repetitions, consistency) -> ``supp.partition``; group BMC frequencies and PXP -> ``bms``; numbers of the text that
no table holds -> ``text``.

Run via ``scripts/exploratory/stats/run_paper_tables.py``; output goes to
``results/exploratory/merged/paper_tables/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats

from motinf.exploratory.stats import dz_ci, holm
from motinf.exploratory.variance_partition import RHO_BOUNDS
from motinf.prereg.rm_anova_exact import rm_anova_exact

SEGS = ("exp1", "exp2")
SAMPLES = {"exp1": ("exp1",), "exp2": ("exp2",), "merged": ("exp1", "exp2")}
INC = {"loss": -1.0, "neutral": 0.0, "gain": 1.0}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _pref(df: pd.DataFrame, seg: str) -> pd.DataFrame:
    df = df.copy()
    df["participant"] = seg + "__" + df["participant"].astype(str)
    if "incentive" in df.columns:
        df["incentive"] = df["incentive"].astype(float)
    return df


def load_all(root: Path) -> dict[str, dict[str, pd.DataFrame]]:
    """Per-segment participant-level tables (participant ids prefixed ``seg__``)."""
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for seg in SEGS:
        p = root / "results" / "prereg" / seg
        e = root / "results" / "exploratory" / seg / "variance_partition"
        beh = _pref(pd.read_csv(p / "behavior" / "data_mean_inc.csv"), seg)
        assert not beh.duplicated(["participant", "incentive"]).any(), "data_mean_inc not unique"
        sd_inf = _pref(pd.read_csv(p / "model_fit" / "stats" / "sd_inf_long.csv"), seg)
        sd_inf = sd_inf[["participant", "incentive", "sd_inf"]].copy()
        sd_inf["log_sd_inf"] = np.log(sd_inf["sd_inf"])
        b = root / "results" / "exploratory" / seg / "modeling" / "stats" / "bias_stats" / "params"
        probit = pd.read_csv(b / "probit_carryover_coefs_long.csv").pivot_table(
            index=["participant", "incentive"], columns="coef", values="value", aggfunc="first").reset_index()
        out[seg] = {
            "beh": beh,
            "seq": _pref(pd.read_csv(p / "behavior" / "data_mean_inc_seqlen.csv"), seg),
            "sd_inf": sd_inf,
            "vp_categ": _pref(pd.read_csv(e / "var_partition_categ.csv"), seg),
            "vp_estim": _pref(pd.read_csv(e / "variance_partition_estim.csv"), seg),
            "bias_mad": _pref(pd.read_csv(b / "estimation_mean_abs_dev.csv"), seg),
            "bias_probit": _pref(probit, seg),  # one column per coefficient
            "bias_temporal": _pref(pd.read_csv(b / "temporal_beta_long.csv"), seg),
        }
        for k in ("vp_categ", "vp_estim"):
            vp = out[seg][k]
            assert (vp["bias_sd"] > 0).all(), f"{seg} {k}: non-positive bias SD (partition not bounded?)"
            vp["log_bias_sd"] = np.log(vp["bias_sd"])
    return out


def sample_df(data: dict, sample: str, key: str) -> pd.DataFrame:
    return pd.concat([data[s][key] for s in SAMPLES[sample]], ignore_index=True)


def _at_lower_bound(vp: pd.DataFrame, bound: float = RHO_BOUNDS[0]) -> pd.Series:
    """Cells whose stored bias share bias_var / (noise_var + bias_var) sits at the partition's lower bound."""
    share = vp["bias_sd"] ** 2 / (vp["noise_sd"] ** 2 + vp["bias_sd"] ** 2)
    return pd.Series(np.isclose(share, bound, rtol=1e-3), index=vp.index)


# --------------------------------------------------------------------------- #
# Effect sizes / tests
# --------------------------------------------------------------------------- #


def paired_effect(diff: pd.Series, tail: str = "two-sided") -> dict:
    """One-sample test of a per-participant paired difference.

    Two-sided tests carry the default JZS ``bf10``; one-tailed tests (the pre-registered |incentive| tests of
    Table 1) carry none, as pingouin computes two-sided Bayes factors only and the paper reports no one-tailed BF.
    """
    diff = diff.dropna()
    n = len(diff)
    t = float(diff.mean() / (diff.std(ddof=1) / np.sqrt(n)))
    alt = {"two-sided": "two-sided", "greater": "greater", "less": "less"}[tail]
    p = float(stats.ttest_1samp(diff, 0.0, alternative=alt).pvalue)
    dz, lo, hi = dz_ci(t, n)
    out = {
        "n": n, "mean_diff": float(diff.mean()), "t": t, "df": n - 1, "p": p,
        "dz": dz, "dz_lo": lo, "dz_hi": hi,
    }
    if tail == "two-sided":
        out["bf10"] = float(pg.bayesfactor_ttest(t, n))
    return out


def wide(df: pd.DataFrame, dv: str) -> pd.DataFrame:
    w = df.pivot(index="participant", columns="incentive", values=dv).dropna()
    return w.rename(columns={-1.0: "loss", 0.0: "neutral", 1.0: "gain"})


def contrasts(df: pd.DataFrame, dv: str) -> dict:
    """gain-neutral, loss-neutral, gain-loss (two-sided, Holm over the three)."""
    w = wide(df, dv)
    res = {
        "gain_neutral": paired_effect(w["gain"] - w["neutral"]),
        "loss_neutral": paired_effect(w["loss"] - w["neutral"]),
        "gain_loss": paired_effect(w["gain"] - w["loss"]),
    }
    ph = holm([res[k]["p"] for k in res])
    for k, p in zip(res, ph):
        res[k]["p_holm"] = p
    return res


def abs_effect(df: pd.DataFrame, dv: str, tail: str) -> dict:
    """|incentive| vs neutral: mean(gain, loss) - neutral, one-tailed as pre-registered."""
    w = wide(df, dv)
    return paired_effect((w["gain"] + w["loss"]) / 2 - w["neutral"], tail=tail)


def omnibus(df: pd.DataFrame, dv: str) -> dict:
    """One-factor (incentive) rmANOVA; ``p_gg`` is the GG-corrected p with exact epsilon."""
    r = rm_anova_exact(df[["participant", "incentive", dv]], dv, "incentive").iloc[0]
    n = int(r["n_participants"])
    return {
        "n": n, "F": float(r["F"]), "df1": 2, "df2": 2 * (n - 1),
        "p": float(r["p_unc"]), "p_gg": float(r["p_GG_corr"]),
        "np2": float(r["np2"]), "eps": float(r["eps"]),
    }


def seq_omnibus(df: pd.DataFrame) -> dict:
    """rmANOVA of log choice variability on sequence length x incentive (pre-registered)."""
    a = rm_anova_exact(df[["participant", "seqlen", "incentive", "choice_var_log"]],
                       "choice_var_log", ["seqlen", "incentive"])
    out = {}
    for src, key in [("seqlen", "seqlen"), ("incentive", "incentive"),
                     ("seqlen * incentive", "seqlen_x_incentive")]:
        r = a.loc[a["Source"] == src].iloc[0]
        out[key] = {
            "n": int(r["n_participants"]), "F": float(r["F"]),
            "df1": int(r["ddof1"]), "df2": int(r["ddof2"]),
            "p": float(r["p_unc"]), "p_gg": float(r["p_GG_corr"]),
            "np2": float(r["np2"]), "eps": float(r["eps"]),
        }
    return out


def measure(df: pd.DataFrame, dv: str, abs_tail: str | None = None, desc_dv: str | None = None,
            scale: float = 1.0) -> dict:
    """Omnibus + gain-neutral, loss-neutral, gain-loss (Holm) + pooled incentive-neutral (two-sided).

    ``desc_dv``: column whose per-condition mean (SD) is reported (defaults to ``dv``; use it when the
    test is on a log scale and descriptives should be in native units). ``scale`` multiplies the
    descriptive means/SDs (and is stored so mean differences can be rescaled, e.g. s -> ms).
    """
    w = wide(df, dv)
    out = {"omnibus": omnibus(df, dv), **contrasts(df, dv)}
    out["pooled"] = paired_effect((w["gain"] + w["loss"]) / 2 - w["neutral"])
    wd = wide(df, desc_dv or dv)
    out["desc"] = {k: {"mean": float(wd[k].mean() * scale), "sd": float(wd[k].std(ddof=1) * scale)}
                   for k in ("neutral", "gain", "loss")}
    out["scale"] = scale
    if abs_tail:
        out["abs"] = abs_effect(df, dv, abs_tail)
    return out


def contrast_values(df: pd.DataFrame, dv: str) -> dict[str, pd.Series]:
    """Per-participant contrasts (the quantities tested by :func:`measure`)."""
    w = wide(df, dv)
    return {
        "gain_neutral": w["gain"] - w["neutral"],
        "loss_neutral": w["loss"] - w["neutral"],
        "pooled": (w["gain"] + w["loss"]) / 2 - w["neutral"],
        "gain_loss": w["gain"] - w["loss"],
    }


def experiment_difference(a: pd.Series, b: pd.Series) -> dict:
    """Experiment 1 vs 2 on a per-participant quantity: Welch t-test, Hedges' g (small-sample corrected) with an
    approximate 95% CI, and the default JZS Bayes factor of the two-sample (pooled-variance) t."""
    a, b = a.dropna(), b.dropna()
    n1, n2 = len(a), len(b)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    v1, v2 = a.var(ddof=1) / n1, b.var(ddof=1) / n2
    welch_df = float((v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1)))
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    d = float((a.mean() - b.mean()) / sp)
    g = d * (1 - 3 / (4 * (n1 + n2) - 9))
    se_g = float(np.sqrt((n1 + n2) / (n1 * n2) + g ** 2 / (2 * (n1 + n2))))
    t_student = d / np.sqrt(1 / n1 + 1 / n2)
    return {
        "n1": n1, "n2": n2, "t": float(t), "df": welch_df, "p": float(p),
        "g": g, "g_lo": g - 1.96 * se_g, "g_hi": g + 1.96 * se_g,
        "bf10": float(pg.bayesfactor_ttest(t_student, n1, n2)),
    }


# Measures of the main-text Tables 2-3: key in ``table2`` -> (participant-level table, tested column)
TABLE23_MEASURES = {
    "sensitivity": ("beh", "sensitivity"),
    "categ_optim": ("beh", "optim"),
    "log_sd_inf": ("sd_inf", "log_sd_inf"),
    "estim_sd": ("beh", "estim_sd"),
    "categ_noise_log": ("vp_categ", "noise_sd_log"),
    "categ_bias": ("vp_categ", "log_bias_sd"),
    "estim_noise_log": ("vp_estim", "noise_sd_log"),
    "estim_bias": ("vp_estim", "log_bias_sd"),
}
BIAS_PROBIT = ("b_0", "b_prev_resp", "b_prev_stim", "b_prev_key")


def bias_measures(data: dict, sample: str) -> dict:
    """Common response biases, from the per-participant outputs of ``run_bias_stats.py``.

    Estimation: mean absolute cardinal and serial bias (degrees; kernel-smoothing regression). Integration: overall
    evidence weighting (temporal weights averaged over sample positions; its incentive effect equals the incentive
    main effect of the incentive x position rmANOVA) and the signed probit carryover coefficients. ``temporal`` holds
    that two-way rmANOVA (exact Greenhouse-Geisser epsilon).
    """
    mad = sample_df(data, sample, "bias_mad")
    probit = sample_df(data, sample, "bias_probit")
    temp = sample_df(data, sample, "bias_temporal")
    out = {task: measure(mad[mad["task"] == task], "mean_abs_dev") for task in ("cardinal", "serial")}
    weights = temp.groupby(["participant", "incentive"], as_index=False)["beta"].mean()
    out["evidence_weight"] = measure(weights, "beta")
    for coef in BIAS_PROBIT:
        out[coef] = measure(probit, coef)
    a = rm_anova_exact(temp[["participant", "incentive", "position", "beta"]], "beta", ["incentive", "position"])
    out["temporal"] = {}
    for src, key in (("incentive", "incentive"), ("incentive * position", "incentive_x_position")):
        r = a.loc[a["Source"] == src].iloc[0]
        out["temporal"][key] = {
            "n": int(r["n_participants"]), "F": float(r["F"]), "df1": int(r["ddof1"]), "df2": int(r["ddof2"]),
            "p": float(r["p_unc"]), "p_gg": float(r["p_GG_corr"]), "np2": float(r["np2"]), "eps": float(r["eps"]),
        }
    return out


def choice_var_natural(root: Path) -> dict:
    """Natural-scale choice variability, sequence length x incentive rmANOVA per experiment (participants with extreme
    variance excluded), read from the behaviour pipeline (``behavior_stats``, exact epsilon)."""
    out = {}
    for seg in SEGS:
        f = root / "results" / "exploratory" / seg / "behavior/stats/behav_extra/params" / "rm_anova_choice_var_natural_seqlen_x_incentive.csv"
        a = pd.read_csv(f)
        assert "p_GG_corr" in a.columns and "n_participants" in a.columns, f"not an exact-epsilon table: {f}"
        out[seg] = {}
        for src, key in (("seqlen", "seqlen"), ("incentive", "incentive"), ("seqlen * incentive", "seqlen_x_incentive")):
            r = a.loc[a["Source"] == src].iloc[0]
            out[seg][key] = {
                "n": int(r["n_participants"]), "F": float(r["F"]), "df1": int(r["ddof1"]), "df2": int(r["ddof2"]),
                "p": float(r["p_unc"]), "p_gg": float(r["p_GG_corr"]), "np2": float(r["np2"]), "eps": float(r["eps"]),
            }
    return out


QUEST_DIR = Path("results/exploratory/merged/modeling/stats/questionnaire_model_correlations")


def questionnaire_rlm(root: Path) -> dict:
    """Robust (Huber) regressions of model parameters on questionnaire subscales and on demographics (merged sample).

    One regression per parameter x predictor block (BIS/BAS; SPSRQ-RC, Experiment 2 only; age and sex): coefficient,
    SE, z and p per predictor, R^2 and n per regression.
    """
    out = {}
    for kind, parts in (("inference_only", ("simple_model/core/rlm", "simple_model/core/demographics/rlm")),
                        ("full", ("full_model/rlm", "full_model/demographics/rlm"))):
        d = pd.concat([pd.read_csv(root / QUEST_DIR / p / "rlm_coefs_long.csv") for p in parts], ignore_index=True)
        coefs: dict = {}
        fits: dict = {}
        for _, r in d.iterrows():
            coefs.setdefault(r["parameter"], {})[r["predictor"]] = {
                "coef": float(r["coef"]), "se": float(r["se"]), "z": float(r["z"]), "p": float(r["pvalue"]),
                "block": r["block"],
            }
            fits.setdefault(r["parameter"], {})[r["block"]] = {"r2": float(r["r2_full"]), "n": int(r["n"])}
        out[kind] = {"coefs": coefs, "fits": fits}
    return out


def choice_var_prediction_rlm(root: Path) -> dict:
    """Full-model noise (sigma) and incentive-modulation (delta) parameters predicting each participant's
    choice-variance midpoint and slope, and their incentive effects (robust regression, merged sample)."""
    d = pd.read_csv(root / "results/exploratory/merged/modeling/stats/full_model_stats/correlations/rlm_coefs.csv")
    d = d[d["predictor"] != "const"]
    out: dict = {}
    for (block, y), g in d.groupby(["block", "y"]):
        out.setdefault(block, {})[y] = {
            "r2": float(g["r2_full"].iloc[0]), "n": int(g["n"].iloc[0]),
            "coefs": {r["predictor"]: {"coef": float(r["coef"]), "se": float(r["se"]), "z": float(r["z"]),
                                       "p": float(r["pvalue"])} for _, r in g.iterrows()},
        }
    return out


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #


def bms(root: Path) -> dict:
    """Group BMC results (inference model frequency and PXP) per sample."""
    def rd(p): return pd.read_csv(root / p)
    paths = {
        "exp1": ("results/prereg/exp1/model_fit/stats/groupbmc_incentive_effects.csv",
                 "results/prereg/exp1/model_fit/stats/groupbmc_noise_per_incentive.csv"),
        "exp2": ("results/prereg/exp2/model_fit/stats/groupbmc_incentive_effects.csv",
                 "results/prereg/exp2/model_fit/stats/groupbmc_noise_per_incentive.csv"),
        "merged": ("results/exploratory/merged/modeling/stats/model_fit_stats/params/groupbmc_incentive_effects.csv",
                   "results/exploratory/merged/modeling/stats/model_fit_stats/params/groupbmc_noise_per_incentive.csv"),
    }
    out = {}
    for s, (pm, pn) in paths.items():
        m, n = rd(pm), rd(pn)
        mi = m.loc[m["model"] == "inf_v"].iloc[0]
        ni = n.loc[n["model"] == "inf"]
        out[s] = {
            "n": int(mi["n"]),
            "mod_pxp": float(mi["pxp"]), "mod_freq": float(mi["frequency_mean"]),
            "noise_pxp_min": float(ni["pxp"].min()),
            "noise_freq_min": float(ni["frequency_mean"].min()),
            "noise_freq_max": float(ni["frequency_mean"].max()),
        }
    return out


def rt_evidence(rt_dir: Path) -> dict:
    """RT ~ evidence (Integration task) from the pipeline outputs of ``rt_regression.py``.

    Per participant x incentive x sequence length, log RT was regressed on the evidence supporting the
    chosen option (``ev_chosen``, in log-likelihood-ratio units). The per-cell intercept (log RT at zero
    evidence) and slope (log RT per unit evidence) are tested against zero (slope) and, per sequence
    length, for the incentive - neutral effect (Holm over the six tests per coefficient). Values are read
    from ``rt_evidence_cell_linear_tests.csv`` / ``_coefs.csv`` (n = 216); percentages are
    100 * (exp(effect) - 1).
    """
    tests = pd.read_csv(rt_dir / "rt_evidence_cell_linear_tests.csv")
    coefs = pd.read_csv(rt_dir / "rt_evidence_cell_linear_coefs.csv")
    keep = ["mean_diff", "pct_rt", "t", "df", "p", "dz", "dz_lo", "dz_hi", "p_holm", "n"]
    out: dict = {"slope_vs_zero": {}, "incentive_vs_neutral": {}, "baseline_rt_zero_evidence_ms": {}}
    for _, r in tests[tests["family"] == "slope_vs_zero"].iterrows():
        out["slope_vs_zero"].setdefault(str(r["incentive"]), {})[str(int(r["seqlen"]))] = {k: float(r[k]) for k in keep}
    for _, r in tests[tests["family"] == "incentive_vs_neutral"].iterrows():
        key = f"{r['coef']}|{r['incentive']}"
        out["incentive_vs_neutral"].setdefault(key, {})[str(int(r["seqlen"]))] = {k: float(r[k]) for k in keep}
    neutral = coefs[coefs["incentive"] == 0.0]
    for sl, g in neutral.groupby("seqlen"):
        out["baseline_rt_zero_evidence_ms"][str(int(sl))] = float(np.exp(g["intercept"].mean()) * 1000.0)
    an = pd.read_csv(rt_dir / "rt_evidence_anova.csv")
    r = an[an["Source"] == "incentive * ev_bin"].iloc[0]
    out["bin_model_incentive_x_evidence"] = {
        "F": float(r["F"]), "df1": int(r["ddof1"]), "df2": int(r["ddof2"]),
        "p_gg": float(r["p-GG-corr"]), "eps": float(r["eps"]), "n": int(r["n_participants"]),
    }
    return out


def text_numbers(data: dict) -> dict:
    """Numbers quoted in the text that no table holds.

    ``estim_sd_pooled``: reproduction error SD averaged over the three conditions per participant, mean and SEM over
    participants (both experiments). ``power_d_av_exp1``: the Experiment 1 effect the sample size of Experiment 2 was
    chosen for, H3 (mean of gain and loss vs neutral evidence sensitivity) standardized by the average SD of the two
    (d_av, pingouin's paired Cohen d, as in the pre-registered output ``behav_stats.md``).
    """
    w = wide(sample_df(data, "merged", "beh"), "estim_sd").mean(axis=1)
    s = wide(sample_df(data, "exp1", "beh"), "sensitivity")
    inc = (s["gain"] + s["loss"]) / 2
    return {
        "estim_sd_pooled": {"mean": float(w.mean()), "sem": float(w.std(ddof=1) / np.sqrt(len(w))), "n": len(w)},
        "power_d_av_exp1": float(pg.compute_effsize(inc, s["neutral"], paired=True, eftype="cohen")),
    }


def build_all(root: Path) -> dict:
    data = load_all(root)
    res: dict = {"bms": bms(root), "table1": {}, "table2": {}, "supp": {}, "text": text_numbers(data)}

    for s in SAMPLES:
        beh = sample_df(data, s, "beh")
        seq = sample_df(data, s, "seq")
        sdi = sample_df(data, s, "sd_inf")
        vpc = sample_df(data, s, "vp_categ")
        vpe = sample_df(data, s, "vp_estim")

        # ---- Table 1 (confirmatory) ----
        t1 = {
            "estim_sd": {"omnibus": omnibus(beh, "estim_sd"),
                         "abs": abs_effect(beh, "estim_sd", "less")},
            "sensitivity": {"omnibus": omnibus(beh, "sensitivity"),
                            "abs": abs_effect(beh, "sensitivity", "greater")},
            "choice_var": seq_omnibus(seq),
            "log_sd_inf": {"omnibus": omnibus(sdi, "log_sd_inf"),
                           "abs": abs_effect(sdi, "log_sd_inf", "less")},
        }
        res["table1"][s] = t1

        # ---- Table 2 (exploratory contrasts) ----
        vpe = vpe.assign(rho=vpe["rho_testretest"])
        res["table2"][s] = {
            "estim_sd": measure(beh, "estim_sd"),
            # Estimation partition SDs are stored in radians; report descriptives in degrees
            "estim_noise_log": measure(vpe, "noise_sd_log", desc_dv="noise_sd", scale=180.0 / np.pi),
            "estim_bias": measure(vpe, "log_bias_sd", desc_dv="bias_sd", scale=180.0 / np.pi),
            "estim_rt": measure(beh, "rt_estim_median", scale=1000.0),
            "sensitivity": measure(beh, "sensitivity"),
            "categ_optim": measure(beh, "optim", scale=100.0),
            "log_sd_inf": measure(sdi, "log_sd_inf", desc_dv="sd_inf"),
            "categ_noise_log": measure(vpc, "noise_sd_log", desc_dv="noise_sd"),
            "categ_bias": measure(vpc, "log_bias_sd", desc_dv="bias_sd"),
            "categ_rt": measure(beh, "rt_median", scale=1000.0),
        }

        # ---- Supplement: extra partition measures + sensitivity variants ----
        res["supp"].setdefault("partition", {})[s] = {
            "estim_rho": measure(vpe, "rho"),
            "categ_consistency": measure(vpc, "consistency"),
            "categ_bias_share": measure(vpc, "rho"),
            # natural-scale bias SD (Table 3 note; the paper tests log bias SD)
            "estim_bias_natural": measure(vpe, "bias_sd", scale=180.0 / np.pi),
            "categ_bias_natural": measure(vpc, "bias_sd"),
            "estim_noise_sd": measure(vpe, "noise_sd", scale=180.0 / np.pi),
            "categ_noise_sd": measure(vpc, "noise_sd"),
            # cells whose bias share sits at the lower bound (0.001) of the stored partition
            "n_bias_share_at_bound_estim": {k: int(_at_lower_bound(vpe[vpe.incentive == v]).sum())
                                            for k, v in INC.items()},
            "n_bias_share_at_bound_categ": {k: int(_at_lower_bound(vpc[vpc.incentive == v]).sum())
                                            for k, v in INC.items()},
            "n_cells": int(vpc["participant"].nunique()),
        }

        # ---- Estimation error SD on the log scale (Supplementary Table 1) ----
        res["supp"].setdefault("estim_sd_log", {})[s] = omnibus(
            beh.assign(log_estim=np.log(beh["estim_sd"])), "log_estim"
        )

        # ---- Common response biases ----
        res["supp"].setdefault("biases", {})[s] = bias_measures(data, s)

    # ---- Experiment 1 vs 2, for every contrast of every Table 2-3 measure (same test scale as the tables) ----
    het = {}
    for key, (dk, dv) in TABLE23_MEASURES.items():
        c1, c2 = contrast_values(data["exp1"][dk], dv), contrast_values(data["exp2"][dk], dv)
        h = {k: experiment_difference(c1[k], c2[k]) for k in c1}
        three = ("gain_neutral", "loss_neutral", "gain_loss")
        for k, p in zip(three, holm([h[k]["p"] for k in three])):
            h[k]["p_holm"] = p
        het[key] = h
    res["supp"]["heterogeneity"] = het
    res["supp"]["choice_var_natural"] = choice_var_natural(root)
    res["supp"]["questionnaire"] = questionnaire_rlm(root)
    res["supp"]["choice_var_prediction"] = choice_var_prediction_rlm(root)

    # ---- Reaction-time control analyses (merged) ----
    rt = root / "results" / "exploratory" / "merged" / "behavior" / "stats" / "rt_analysis"
    fs = {}
    for name, f in [("categ", "split_effects_sd_inf_log.csv"), ("estim", "split_effects_log_sd_estim.csv")]:
        d = pd.read_csv(rt / f)
        w = d.pivot_table(index=["participant", "contrast"], columns="rt_bin", values="effect").reset_index()
        w["fast_minus_slow"] = w["fast"] - w["slow"]
        g = w.pivot(index="participant", columns="contrast", values="fast_minus_slow").dropna()
        out = {
            "gain": paired_effect(g["gain_minus_neutral"]),
            "loss": paired_effect(g["loss_minus_neutral"]),
            "gain_loss": paired_effect(g["gain_minus_neutral"] - g["loss_minus_neutral"]),
        }
        ph = holm([out[k]["p"] for k in ("gain", "loss", "gain_loss")])
        for k, p in zip(("gain", "loss", "gain_loss"), ph):
            out[k]["p_holm"] = p
        fs[name] = out
    res["supp"]["fast_slow"] = fs

    # noise fast - slow within each incentive, per participant vs 0 (Holm over the three levels)
    fs_inc = {}
    for name, f in [("categ", "diff_fast_minus_slow_sd_inf_log.csv"), ("estim", "diff_fast_minus_slow_log_sd_estim.csv")]:
        d = pd.read_csv(rt / f)
        labels = {-1.0: "loss", 0.0: "neutral", 1.0: "gain"}
        out = {labels[float(inc)]: paired_effect(g.set_index("participant")["diff"]) for inc, g in d.groupby("incentive")}
        keys = ("loss", "neutral", "gain")
        for k, p in zip(keys, holm([out[k]["p"] for k in keys])):
            out[k]["p_holm"] = p
        fs_inc[name] = out
    res["supp"]["fast_slow_by_incentive"] = fs_inc

    corr = {}
    for name, f in [("categ", "corr_input_rt_median_vs_sd_inf_log.csv"),
                    ("estim", "corr_input_rt_estim_vs_log_sd_estim.csv")]:
        d = pd.read_csv(rt / f)
        corr[name] = {}
        for c, g in d.groupby("contrast"):
            x, y = g["effect_rt"].to_numpy(), g["effect_noise"].to_numpy()
            rho = stats.spearmanr(x, y)
            r_full = stats.pearsonr(x, y)

            def mz(v):
                mad = 1.4826 * np.median(np.abs(v - np.median(v)))
                return np.abs(v - np.median(v)) / mad

            keep = (mz(x) < 4) & (mz(y) < 4)
            r_trim = stats.pearsonr(x[keep], y[keep])
            n_t = int(keep.sum())
            corr[name][c] = {
                "n": len(x), "spearman": float(rho[0]), "spearman_p": float(rho[1]),
                "pearson_full": float(r_full[0]), "pearson_full_p": float(r_full[1]),
                "n_trim": n_t, "pearson_trim": float(r_trim[0]), "pearson_trim_p": float(r_trim[1]),
                "bf10_trim": float(pg.bayesfactor_pearson(r_trim[0], n_t)),
            }
    res["supp"]["rt_corr"] = corr
    res["supp"]["rt_evidence"] = rt_evidence(rt)
    return res


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def run_paper_tables(root: Path) -> Path:
    out_dir = root / "results" / "exploratory" / "merged" / "paper_tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = _clean(build_all(root))
    path = out_dir / "paper_tables.json"
    path.write_text(json.dumps(res, indent=1))
    return path
