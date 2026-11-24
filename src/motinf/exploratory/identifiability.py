"""Post-processing shared by the two model-recovery (identifiability) scripts.

Per replicate, group Bayesian model selection on ``-0.5 * AICc`` picks the recovered model (highest expected
frequency; ``replicate_bmc_row``). Over replicates, ``write_confusion_outputs`` writes the confusion matrix (counts and
percent) and the mean protected exceedance probabilities; ``summary_tail`` gives the closing lines of
``run_summary.txt``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from motinf.exploratory.stats import group_bmc


def replicate_bmc_row(gen_name: str, rep: int, aicc: np.ndarray, model_names: list[str]) -> dict:
    """GroupBMC of one replicate: ``aicc`` is participants x candidate models, complete cases only. Invalid
    (``valid_replicate`` False) with fewer than 2 complete participants or when GroupBMC fails."""
    mask = np.all(np.isfinite(aicc), axis=1)
    n_sub = int(mask.sum())
    row = {
        "generative_model": gen_name,
        "replicate": rep,
        "n_participants": n_sub,
        "valid_replicate": False,
        "recovered_model": None,
        "frequency_mean_json": None,
        "pxp_json": None,
    }
    if n_sub < 2:
        return row
    try:
        bmc = group_bmc(-0.5 * aicc[mask, :].T)  # models x participants
    except Exception:
        return row
    freq = bmc["frequency_mean"]
    pxp = bmc["protected_exceedance_probability"]
    return {
        **row,
        "valid_replicate": True,
        "recovered_model": model_names[int(np.nanargmax(freq))],
        "frequency_mean_json": json.dumps(freq.tolist()),
        "pxp_json": json.dumps(pxp.tolist()),
    }


def write_confusion_outputs(rep_df: pd.DataFrame, model_names: list[str], out_dir: Path) -> tuple[pd.DataFrame, list]:
    """Write ``confusion_matrix_{counts,percent}.csv`` and, if any replicate is valid,
    ``pxp_mean_across_replicates.csv`` and ``pxp_mean_matrix.csv``. Returns the percent matrix and the paths."""
    valid = rep_df[rep_df["valid_replicate"] & rep_df["recovered_model"].notna()].copy()
    cnt = (
        valid.groupby(["generative_model", "recovered_model"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=model_names, columns=model_names, fill_value=0)
    )
    cnt_df = cnt.copy()
    cnt_df.insert(0, "generative_model", cnt_df.index)
    paths = [out_dir / "confusion_matrix_counts.csv", out_dir / "confusion_matrix_percent.csv"]
    cnt_df.to_csv(paths[0], index=False)

    pct = cnt.div(cnt.sum(axis=1).replace(0, np.nan), axis=0) * 100.0
    pct_df = pct.copy()
    pct_df.insert(0, "generative_model", pct_df.index)
    pct_df.to_csv(paths[1], index=False)

    # Mean (and std) of the PXP vectors across valid replicates, per generative model.
    pxp_rows = []
    for gen_name in model_names:
        stacks = [np.asarray(json.loads(s), dtype=float) for s in valid.loc[valid["generative_model"] == gen_name,
                                                                          "pxp_json"]]
        if not stacks:
            continue
        mat = np.vstack(stacks)
        pxp_rows.append(
            {
                "generative_model": gen_name,
                **{f"pxp_{m}": float(np.nanmean(mat[:, j])) for j, m in enumerate(model_names)},
                **{f"pxp_std_{m}": float(np.nanstd(mat[:, j], ddof=0)) for j, m in enumerate(model_names)},
                "n_replicates_used": int(len(stacks)),
            }
        )
    if pxp_rows:
        pxp_agg = pd.DataFrame(pxp_rows)
        paths.append(out_dir / "pxp_mean_across_replicates.csv")
        pxp_agg.to_csv(paths[-1], index=False)
        mean_only = pxp_agg.set_index("generative_model")[[f"pxp_{m}" for m in model_names]].rename(
            columns={f"pxp_{m}": m for m in model_names}
        )
        mean_only = mean_only.reindex(index=model_names)
        mean_only.insert(0, "generative_model", mean_only.index)
        paths.append(out_dir / "pxp_mean_matrix.csv")
        mean_only.to_csv(paths[-1], index=False)
    return pct, paths


def summary_tail(rep_df: pd.DataFrame, pct: pd.DataFrame, model_names: list[str], *, fit_fail_count: int,
                 skip_participant_count: int, row_sums_heading: str) -> list[str]:
    """The closing lines of ``run_summary.txt``: replicate counts, failures and the row sums of the percent matrix."""
    lines = [
        "",
        f"total_replicates: {len(rep_df)}",
        f"valid_replicates: {int(rep_df['valid_replicate'].sum())}",
        f"invalid_replicates: {int((~rep_df['valid_replicate']).sum())}",
        f"fit_fail_count: {fit_fail_count}",
        f"skip_participant_count: {skip_participant_count}",
        "",
        row_sums_heading,
    ]
    row_sums = pct.sum(axis=1)
    for m in model_names:
        v = row_sums.get(m, np.nan)
        lines.append(f"  {m}: {v:.2f}" if np.isfinite(v) else f"  {m}: nan")
    return lines
