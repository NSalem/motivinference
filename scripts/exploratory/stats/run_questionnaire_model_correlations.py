"""Robust regressions of model parameters on questionnaire subscales and demographics.

Participant-level parameters of
- the single-noise inference model (prereg per-condition fits: neutral sigma_inf and the gain/loss
  log-noise differences, plus the estimation SD and its incentive differences), and
- the full model (``fit_full_model.npz``: sigma_inf, sigma_sel, delta_inf, delta_sel, fixed estimation
  SD and delta),
are each regressed (Huber RLM, z-scored predictors) on one predictor block at a time: BIS/BAS
(BIS, Drive, Reward, Fun); SPSRQ-RC sensitivity to punishment/reward (Experiment 2 only); and, when
the questionnaire file has age and sex, ``z(age) + sex_male``.

Outputs (``results/exploratory/<exp>/modeling/stats/questionnaire_model_correlations/``):
    simple_model/core/rlm/rlm_coefs_long.csv, simple_model/core/demographics/rlm/rlm_coefs_long.csv
    full_model/rlm/rlm_coefs_long.csv, full_model/demographics/rlm/rlm_coefs_long.csv
read by ``paper_tables.questionnaire_rlm`` (Supp. Tables 3 and 5) and ``scripts/exploratory/figures/sfig3.py``.

Usage (from repo root):
    python scripts/exploratory/stats/run_questionnaire_model_correlations.py --exp-name merged
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

from motinf.exploratory.full_model_results import load_full_model_params_for_segments
from motinf.exploratory.full_model_fitting import (
    build_estimation_ang_params_table,
)
from motinf.exploratory.cohorts import (
    cohort_subexperiments,
    merge_data_clean_est,
    merge_sd_inf_long,
    participant_prefixed,
)
from motinf.exploratory.rlm_regression import fit_rlm_block

CORE_QUESTIONNAIRE_MAP: dict[str, str] = {
    "bis": "bis_score",
    "bas_drv": "bas_drv_score",
    "bas_rew": "bas_rew_score",
    "bas_fun": "bas_fun_score",
}

# Present in exp2 only; exp1 rows are left as NaN when pooled.
OPTIONAL_QUESTIONNAIRE_MAP: dict[str, str] = {
    "spsrqrc_sp": "spsrqrc_sp_sum",
    "spsrqrc_sr": "spsrqrc_sr_sum",
}

QUESTIONNAIRE_MAP: dict[str, str] = {
    **CORE_QUESTIONNAIRE_MAP,
    **OPTIONAL_QUESTIONNAIRE_MAP,
}

SIMPLE_PARAMS: list[str] = [
    "sd_ang",
    "sd_inf",
    "motiv_ang_g",
    "motiv_ang_l",
    "motiv_inf_g",
    "motiv_inf_l",
]

FULL_PARAMS: list[str] = [
    "sd_ang",
    "sd_inf",
    "sd_sel",
    "motiv_ang",
    "motiv_inf",
    "motiv_sel",
]

BIS_BAS_SUBSCALES: list[str] = list(CORE_QUESTIONNAIRE_MAP.keys())
SPSRQ_SUBSCALES: list[str] = list(OPTIONAL_QUESTIONNAIRE_MAP.keys())

DEMOGRAPHIC_PREDICTORS: list[str] = ["age", "sex_male"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Robust regressions of model parameters on questionnaire subscales and demographics."
    )
    parser.add_argument(
        "--exp-name",
        default="merged",
        help="exp1, exp2 or merged (pools exp1+exp2; see motinf.exploratory.cohorts).",
    )
    return parser.parse_args()


def _encode_sex_male(series: pd.Series) -> pd.Series:
    """Encode sex as 0=Female, 1=Male; unmapped values become NaN."""
    s = series.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=series.index, dtype=float)
    out[s.isin(["male", "m"])] = 1.0
    out[s.isin(["female", "f"])] = 0.0
    return out


def _demographics_available(q: pd.DataFrame) -> bool:
    return (
        "age" in q.columns
        and "sex_male" in q.columns
        and q["age"].notna().any()
        and q["sex_male"].notna().any()
    )


def _load_questionnaire_for_segments(segments: list[str]) -> pd.DataFrame:
    prefix_pids = len(segments) > 1
    chunks: list[pd.DataFrame] = []
    for seg in segments:
        path = ROOT / "data" / "raw" / seg / "questionnaire" / "questionnaire_data.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing questionnaire file: {path}")

        df = pd.read_csv(path)
        required = ["participant", *CORE_QUESTIONNAIRE_MAP.values()]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"Questionnaire file missing required columns: {missing}")

        cols = ["participant", *CORE_QUESTIONNAIRE_MAP.values()]
        for src_col in OPTIONAL_QUESTIONNAIRE_MAP.values():
            if src_col in df.columns:
                cols.append(src_col)
        for demo_col in ("age", "sex"):
            if demo_col in df.columns:
                cols.append(demo_col)

        out = df[cols].copy()
        out["participant"] = out["participant"].astype(str)
        if prefix_pids:
            out["participant"] = out["participant"].map(lambda p, s=seg: participant_prefixed(s, p))

        for src_col in CORE_QUESTIONNAIRE_MAP.values():
            out[src_col] = pd.to_numeric(out[src_col], errors="coerce")
        for src_col in OPTIONAL_QUESTIONNAIRE_MAP.values():
            if src_col not in out.columns:
                out[src_col] = np.nan
            else:
                out[src_col] = pd.to_numeric(out[src_col], errors="coerce")

        rename_map = {v: k for k, v in QUESTIONNAIRE_MAP.items() if v in out.columns}
        chunk = out.rename(columns=rename_map)
        if "sex" in chunk.columns:
            chunk["sex_male"] = _encode_sex_male(chunk["sex"])
        if "age" in chunk.columns:
            chunk["age"] = pd.to_numeric(chunk["age"], errors="coerce")
        chunks.append(chunk)

    merged = pd.concat(chunks, ignore_index=True)
    merged = merged.drop_duplicates(subset=["participant"], keep="first").reset_index(drop=True)
    for key in OPTIONAL_QUESTIONNAIRE_MAP:
        if key not in merged.columns:
            merged[key] = np.nan
    if "age" not in merged.columns:
        merged["age"] = np.nan
    if "sex_male" not in merged.columns:
        merged["sex_male"] = np.nan
    return merged


def _assert_three_incentives(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    d = df.copy()
    d["incentive"] = pd.to_numeric(d["incentive"], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d["participant"] = d["participant"].astype(str)

    d = d[d["incentive"].isin([-1.0, 0.0, 1.0])]
    counts = d.groupby("participant", observed=True)["incentive"].nunique(dropna=True)
    keep = counts[counts == 3].index
    return d[d["participant"].isin(keep)].copy()


def _build_simple_params_for_segments(segments: list[str]) -> pd.DataFrame:
    """Neutral sigma_inf and log-noise differences (neutral - gain, neutral - loss), plus estimation SD/deltas."""
    df_inf = merge_sd_inf_long(ROOT, segments)
    df_est = merge_data_clean_est(ROOT, segments)
    req_inf = {"participant", "incentive", "sd_inf_log"}
    if not req_inf.issubset(df_inf.columns):
        raise KeyError(f"sd_inf table missing required columns: {sorted(req_inf)}")
    df_inf = _assert_three_incentives(df_inf[["participant", "incentive", "sd_inf_log"]], "sd_inf_log")
    p_inf = df_inf.pivot_table(
        index="participant",
        columns="incentive",
        values="sd_inf_log",
        aggfunc="first",
    )
    p_inf.columns = [float(c) for c in p_inf.columns]
    p_inf = p_inf.rename(columns={-1.0: "sd_inf_log_l", 0.0: "sd_inf_log_0", 1.0: "sd_inf_log_g"})
    p_inf = p_inf.reset_index()
    p_inf["sd_inf"] = np.exp(p_inf["sd_inf_log_0"])
    p_inf["motiv_inf_g"] = p_inf["sd_inf_log_0"] - p_inf["sd_inf_log_g"]
    p_inf["motiv_inf_l"] = p_inf["sd_inf_log_0"] - p_inf["sd_inf_log_l"]

    p_ang = build_estimation_ang_params_table(df_est)
    df_simple = p_ang.merge(
        p_inf[["participant", "sd_inf", "motiv_inf_g", "motiv_inf_l"]],
        on="participant",
        how="inner",
    )
    return df_simple[["participant", *SIMPLE_PARAMS]].copy()


def _merge_and_log(name: str, q: pd.DataFrame, pars: pd.DataFrame, param_cols: list[str]) -> pd.DataFrame:
    p = pars[["participant", *param_cols]].copy()
    m = q.merge(p, on="participant", how="inner")
    n_q, n_p, n_m = q["participant"].nunique(), p["participant"].nunique(), m["participant"].nunique()
    print(f"[{name}] participants: questionnaire={n_q}, params={n_p}, merged={n_m}")
    if n_m == 0:
        raise RuntimeError(f"[{name}] No overlapping participants after merge.")
    return m


def _rlm_rows(
    merged_df: pd.DataFrame,
    *,
    model_kind: str,
    param_cols: list[str],
    blocks: dict[str, list[str]],
    zscore_cols: list[str] | None = None,
) -> pd.DataFrame:
    """One RLM per parameter x block (intercept not reported); ``zscore_cols=None`` z-scores every predictor."""
    rows: list[dict] = []
    for param in param_cols:
        for block_name, x_cols in blocks.items():
            fit = fit_rlm_block(merged_df, param, x_cols, zscore_cols=zscore_cols)
            for row in fit["rows"]:
                if row["predictor"] == "const":
                    continue
                rows.append(
                    {
                        "model_kind": model_kind,
                        "parameter": param,
                        "block": block_name,
                        "predictor": row["predictor"],
                        "coef": row["coef"],
                        "se": row["se"],
                        "z": row["z"],
                        "pvalue": row["pvalue"],
                        "n": fit["n"],
                        "r2_full": fit["r2_full"],
                    }
                )
    return pd.DataFrame(rows)


def _rlm_blocks_available(merged_df: pd.DataFrame) -> dict[str, list[str]]:
    blocks: dict[str, list[str]] = {"bis_bas": BIS_BAS_SUBSCALES}
    if any(col in merged_df.columns and merged_df[col].notna().any() for col in SPSRQ_SUBSCALES):
        blocks["spsrq"] = SPSRQ_SUBSCALES
    return blocks


def _write_rlm(
    merged_df: pd.DataFrame,
    *,
    model_kind: str,
    param_cols: list[str],
    results_dir: Path,
    with_demographics: bool,
) -> None:
    """Questionnaire-block and demographic RLM tables."""
    demo_coef_df = None
    if with_demographics:
        demo_coef_df = _rlm_rows(
            merged_df,
            model_kind=model_kind,
            param_cols=param_cols,
            blocks={"demographics": DEMOGRAPHIC_PREDICTORS},
            zscore_cols=["age"],
        )
        (results_dir / "demographics" / "rlm").mkdir(parents=True, exist_ok=True)
        demo_coef_df.to_csv(results_dir / "demographics" / "rlm" / "rlm_coefs_long.csv", index=False)

    blocks = _rlm_blocks_available(merged_df)
    coef_df = _rlm_rows(merged_df, model_kind=model_kind, param_cols=param_cols, blocks=blocks)
    (results_dir / "rlm").mkdir(parents=True, exist_ok=True)
    coef_df.to_csv(results_dir / "rlm" / "rlm_coefs_long.csv", index=False)


def main() -> None:
    exp_name = parse_args().exp_name
    out_base = ROOT / "results" / "exploratory" / exp_name / "modeling" / "stats" / "questionnaire_model_correlations"

    cohort_segs = cohort_subexperiments(exp_name)
    q = _load_questionnaire_for_segments(cohort_segs)
    with_demographics = _demographics_available(q)
    if not with_demographics:
        print("Demographics: skipped (age/sex not available in questionnaire data)")

    simple = _merge_and_log("simple_model", q, _build_simple_params_for_segments(cohort_segs), SIMPLE_PARAMS)
    full = _merge_and_log("full_model", q, load_full_model_params_for_segments(ROOT, cohort_segs), FULL_PARAMS)

    _write_rlm(
        simple,
        model_kind="simple_core",
        param_cols=SIMPLE_PARAMS,
        results_dir=out_base / "simple_model" / "core",
        with_demographics=with_demographics,
    )
    _write_rlm(
        full,
        model_kind="full_model",
        param_cols=FULL_PARAMS,
        results_dir=out_base / "full_model",
        with_demographics=with_demographics,
    )
    print(f"Wrote tables to: {out_base}")


if __name__ == "__main__":
    main()
