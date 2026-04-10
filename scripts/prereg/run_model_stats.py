"""Preregistered model-fit statistics entrypoint.

Writes CSV tables and `model_stats_summary.md` under `results/prereg/<exp>/model_fit/stats/`.
Figures from those tables: `notebooks/prereg_model_stats_plots.ipynb`.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pingouin as pg

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prereg model statistics.")
    parser.add_argument("--exp-name", default="exp1", help="Experiment subfolder under results/prereg/.")
    return parser.parse_args()


def _npz_to_dict(npz: np.lib.npyio.NpzFile) -> dict[str, Any]:
    return {k: npz[k] for k in npz.files}


def _require_keys(d: dict[str, Any], keys: list[str], source: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise KeyError(f"{source} is missing required keys: {missing}")


def _extract_model_names_from_specs(specs_obj: Any) -> list[str]:
    specs = list(np.asarray(specs_obj, dtype=object))
    names: list[str] = []
    for s in specs:
        if isinstance(s, dict):
            names.append(str(s.get("name", "unknown")))
        else:
            names.append("unknown")
    return names


def _run_group_bmc(log_evidence: np.ndarray) -> dict[str, Any]:
    try:
        from groupBMC.groupBMC import GroupBMC
    except Exception as exc:
        raise ImportError(
            "groupBMC is required. Install/verify import path: from groupBMC.groupBMC import GroupBMC"
        ) from exc
    return GroupBMC(log_evidence).get_result()


def _drop_incomplete_subjects(
    aicc_mat: np.ndarray, participants: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (aicc_complete, participants_complete, participants_dropped)."""
    # aicc_mat: (n_models, n_participants)
    keep = np.all(np.isfinite(aicc_mat), axis=0)
    return aicc_mat[:, keep], participants[keep], participants[~keep]


def _warn_if_dropped(
    label: str,
    n_total: int,
    dropped: np.ndarray,
    *,
    stacklevel: int = 2,
) -> None:
    if dropped.size == 0:
        return
    n_drop = int(dropped.size)
    ids = [str(x) for x in np.asarray(dropped).tolist()]
    warnings.warn(
        f"{label}: excluded {n_drop}/{n_total} participant(s) with incomplete AICc across models "
        f"(missing or non-finite for at least one model): {ids}",
        UserWarning,
        stacklevel=stacklevel,
    )


def _extract_groupbmc_metrics(result: Any, n_models: int) -> dict[str, np.ndarray]:
    # GroupBMCResult fields used directly by contract.
    freq_mean = np.asarray(result.frequency_mean, dtype=float).reshape(-1)
    freq_var = np.asarray(result.frequency_var, dtype=float).reshape(-1)
    xp = np.asarray(result.exceedance_probability, dtype=float).reshape(-1)
    pxp = np.asarray(result.protected_exceedance_probability, dtype=float).reshape(-1)
    if not (len(freq_mean) == len(freq_var) == len(xp) == len(pxp) == n_models):
        raise ValueError("GroupBMC result length mismatch with model count.")
    return {"frequency_mean": freq_mean, "frequency_var": freq_var, "xp": xp, "pxp": pxp}


def _build_per_condition_aicc_matrix_for_incentive(
    per_cond: dict[str, Any], model_names: list[str], incentive_idx: int
) -> np.ndarray:
    participants = np.asarray(per_cond["participants"])
    n_participants = len(participants)
    n_models = len(model_names)
    mat = np.full((n_models, n_participants), np.nan, dtype=float)
    for mi, model_name in enumerate(model_names):
        key = f"aicc_{model_name}"
        if key not in per_cond:
            continue
        model_aicc = np.asarray(per_cond[key], dtype=float)  # (participant, incentive)
        if model_aicc.ndim != 2 or incentive_idx >= model_aicc.shape[1]:
            continue
        mat[mi, :] = model_aicc[:, incentive_idx]
    return mat


def _extract_sd_inf_long(per_cond: dict[str, Any]) -> pd.DataFrame:
    participants = np.asarray(per_cond["participants"])
    incentives = np.asarray(per_cond["incentives"])
    pars_inf = np.asarray(per_cond["pars_inf"], dtype=object)  # (participant, incentive)
    rows: list[dict[str, Any]] = []
    for ip, pid in enumerate(participants):
        for ji, inc in enumerate(incentives):
            p = pars_inf[ip, ji]
            if p is None:
                continue
            p_arr = np.asarray(p, dtype=float).reshape(-1)
            if p_arr.size < 1 or not np.isfinite(p_arr[0]):
                continue
            rows.append({"participant": str(pid), "incentive": float(inc), "sd_inf": float(p_arr[0])})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["sd_inf_log"] = np.log(df["sd_inf"])
    return df


def _paired_ttest_log_sd_inf_abs_1_vs_0(df: pd.DataFrame) -> pd.DataFrame:
    """Paired t-test on log(sd_inf): |incentive| 1 vs 0 (pool -1 and +1), matching prereg behavior."""
    d = df[["participant", "incentive", "sd_inf_log"]].dropna().copy()
    d["incentive_abs"] = np.abs(np.asarray(d["incentive"], dtype=float))
    d["incentive_abs"] = np.round(d["incentive_abs"]).astype(int)
    d = d.groupby(["participant", "incentive_abs"], as_index=False)["sd_inf_log"].mean()
    wide = d.pivot(index="participant", columns="incentive_abs", values="sd_inf_log")
    if 0 not in wide.columns or 1 not in wide.columns:
        return pd.DataFrame(
            {
                "test": ["log(sd_inf): abs incentive 1 vs 0"],
                "error": ["missing abs incentive level 0 or 1 for paired test"],
            }
        )
    t = pg.ttest(wide[1], wide[0], paired=True, alternative="less")
    t.insert(0, "test", "log(sd_inf): abs incentive 1 vs 0")
    return t


def _df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._\n"
    try:
        return df.to_markdown(index=False, floatfmt=".4g") + "\n"
    except ImportError:
        return "```\n" + df.to_string(index=False) + "\n```\n"


def _build_model_stats_markdown(
    *,
    exp_name: str,
    has_per_cond: bool,
    has_inc_eff: bool,
    per_cond_path: Path,
    inc_eff_path: Path,
    df_bmc_per_cond: pd.DataFrame | None,
    df_sd_inf: pd.DataFrame | None,
    anova_sd_inf: pd.DataFrame | None,
    ttest_sd_inf: pd.DataFrame | None,
    df_bmc_inc_eff: pd.DataFrame | None,
) -> str:
    lines: list[str] = [
        "# Preregistered model-fit statistics",
        "",
        f"Experiment: `{exp_name}`",
        "",
    ]

    lines.extend(["## Group BMC — per-condition fits (by incentive)", ""])
    if has_per_cond and df_bmc_per_cond is not None:
        lines.append(_df_to_markdown(df_bmc_per_cond))
    else:
        lines.append(f"_Skipped: missing `{per_cond_path.name}`._")
        lines.append("")

    lines.extend(["## Inferential tests on `log(sd_inf)` (inf-only parameters, per condition)", ""])
    if has_per_cond and df_sd_inf is not None and anova_sd_inf is not None and ttest_sd_inf is not None:
        lines.append("### Repeated-measures ANOVA (`log(sd_inf)` ~ incentive)")
        lines.append("")
        lines.append(_df_to_markdown(anova_sd_inf))
        lines.append(
            "### Paired one-sided t-test (`log(sd_inf)` at |incentive| 1 vs 0; "
            "mean within ±1 pooled; `alternative='less'`)"
        )
        lines.append("")
        lines.append(_df_to_markdown(ttest_sd_inf))
    else:
        lines.append(f"_Skipped: missing `{per_cond_path.name}`._")
        lines.append("")

    lines.extend(["## Group BMC — incentive-effects models", ""])
    if has_inc_eff and df_bmc_inc_eff is not None:
        lines.append(_df_to_markdown(df_bmc_inc_eff))
    else:
        lines.append(f"_Skipped: missing `{inc_eff_path.name}`._")
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    model_fit_dir = ROOT / "results" / "prereg" / args.exp_name / "model_fit"
    stats_dir = model_fit_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    per_cond_path = model_fit_dir / "fit_noise_per_cond.npz"
    inc_eff_path = model_fit_dir / "fit_inc_effects.npz"
    has_per_cond = per_cond_path.exists()
    has_inc_eff = inc_eff_path.exists()
    if not has_per_cond and not has_inc_eff:
        raise FileNotFoundError(
            f"Missing both input files: {per_cond_path} and {inc_eff_path}"
        )

    df_bmc_per_cond: pd.DataFrame | None = None
    df_sd_inf: pd.DataFrame | None = None
    anova_sd_inf: pd.DataFrame | None = None
    ttest_sd_inf: pd.DataFrame | None = None
    df_bmc_inc_eff: pd.DataFrame | None = None

    if has_per_cond:
        per_cond = _npz_to_dict(np.load(str(per_cond_path), allow_pickle=True))
        _require_keys(per_cond, ["participants", "incentives", "categ_model_specs", "pars_inf"], "fit_noise_per_cond.npz")

        # Group BMC for per-condition models, run separately per incentive.
        per_cond_participants = np.asarray(per_cond["participants"])
        incentives = np.asarray(per_cond["incentives"], dtype=float)
        per_cond_model_names = _extract_model_names_from_specs(per_cond["categ_model_specs"])
        rows_bmc_per_cond: list[dict[str, Any]] = []
        for ji, inc in enumerate(incentives):
            per_cond_aicc = _build_per_condition_aicc_matrix_for_incentive(per_cond, per_cond_model_names, ji)
            per_cond_aicc_cc, _, dropped_pc = _drop_incomplete_subjects(
                per_cond_aicc, per_cond_participants
            )
            n_total_pc = int(len(per_cond_participants))
            _warn_if_dropped(
                f"GroupBMC per-condition (incentive={inc})",
                n_total_pc,
                dropped_pc,
                stacklevel=2,
            )
            if per_cond_aicc_cc.shape[1] == 0:
                continue
            n_cc = int(per_cond_aicc_cc.shape[1])
            bmc_res = _run_group_bmc(-0.5 * per_cond_aicc_cc)
            metrics = _extract_groupbmc_metrics(bmc_res, len(per_cond_model_names))
            for mi, model_name in enumerate(per_cond_model_names):
                rows_bmc_per_cond.append(
                    {
                        "incentive": float(inc),
                        "model": model_name,
                        "n": n_cc,
                        "frequency_mean": float(metrics["frequency_mean"][mi]),
                        "frequency_var": float(metrics["frequency_var"][mi]),
                        "xp": float(metrics["xp"][mi]),
                        "pxp": float(metrics["pxp"][mi]),
                    }
                )
        df_bmc_per_cond = pd.DataFrame(rows_bmc_per_cond)
        df_bmc_per_cond.to_csv(stats_dir / "groupbmc_noise_per_incentive.csv", index=False)

        # Inferential stats on sd_inf from per-condition inf-only model.
        df_sd_inf = _extract_sd_inf_long(per_cond)
        anova_sd_inf = pg.rm_anova(
            data=df_sd_inf,
            dv="sd_inf_log",
            within=["incentive"],
            subject="participant",
            detailed=True,
            effsize="np2",
        )
        ttest_sd_inf = _paired_ttest_log_sd_inf_abs_1_vs_0(df_sd_inf)
        df_sd_inf.to_csv(stats_dir / "sd_inf_long.csv", index=False)
        anova_sd_inf.to_csv(stats_dir / "anova_sd_inf_incentive.csv", index=False)
        ttest_sd_inf.to_csv(stats_dir / "ttest_log_sd_inf_abs_1_vs_0.csv", index=False)

    if has_inc_eff:
        inc_eff = _npz_to_dict(np.load(str(inc_eff_path), allow_pickle=True))
        _require_keys(inc_eff, ["participants", "categ_model_specs", "aicc_categ"], "fit_inc_effects.npz")
        # Group BMC for incentive-effects models.
        inc_eff_aicc = np.asarray(inc_eff["aicc_categ"], dtype=float)  # (model, participant)
        inc_eff_model_names = _extract_model_names_from_specs(inc_eff["categ_model_specs"])
        inc_eff_participants = np.asarray(inc_eff["participants"])
        inc_eff_aicc_cc, _, dropped_ie = _drop_incomplete_subjects(
            inc_eff_aicc, inc_eff_participants
        )
        n_total_ie = int(len(inc_eff_participants))
        _warn_if_dropped(
            "GroupBMC incentive-effects",
            n_total_ie,
            dropped_ie,
            stacklevel=2,
        )
        inc_eff_bmc = _run_group_bmc(-0.5 * inc_eff_aicc_cc)
        inc_eff_metrics = _extract_groupbmc_metrics(inc_eff_bmc, len(inc_eff_model_names))
        n_cc_ie = int(inc_eff_aicc_cc.shape[1])
        rows_inc_eff: list[dict[str, Any]] = []
        for mi, model_name in enumerate(inc_eff_model_names):
            rows_inc_eff.append(
                {
                    "model": model_name,
                    "n": n_cc_ie,
                    "frequency_mean": float(inc_eff_metrics["frequency_mean"][mi]),
                    "frequency_var": float(inc_eff_metrics["frequency_var"][mi]),
                    "xp": float(inc_eff_metrics["xp"][mi]),
                    "pxp": float(inc_eff_metrics["pxp"][mi]),
                }
            )
        df_bmc_inc_eff = pd.DataFrame(rows_inc_eff)
        df_bmc_inc_eff.to_csv(stats_dir / "groupbmc_incentive_effects.csv", index=False)

    summary_md = _build_model_stats_markdown(
        exp_name=args.exp_name,
        has_per_cond=has_per_cond,
        has_inc_eff=has_inc_eff,
        per_cond_path=per_cond_path,
        inc_eff_path=inc_eff_path,
        df_bmc_per_cond=df_bmc_per_cond,
        df_sd_inf=df_sd_inf,
        anova_sd_inf=anova_sd_inf,
        ttest_sd_inf=ttest_sd_inf,
        df_bmc_inc_eff=df_bmc_inc_eff,
    )
    out_md = stats_dir / "model_stats_summary.md"
    out_md.write_text(summary_md, encoding="utf-8")
    print(summary_md, end="")


if __name__ == "__main__":
    main()
