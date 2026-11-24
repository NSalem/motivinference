# -*- coding: utf-8 -*-
"""AICc of the full model and of the inference-only models per participant, for their group BMC (Results).

Three already-fitted models are compared on whole-dataset AICc (``-0.5 * AICc`` as log evidence); this script
collects their AICc, and the ``tables`` stage runs the group BMC (``table_stats.full_vs_inference_only_tables``):

1. ``full``: the full model (``fit_full_model.npz``).
2. ``inf_v_prereg``: the pre-registered single-modulation model with the incentive acting on inference
   noise (``inf_v`` in ``results/prereg/<exp>/model_fit/fit_inc_effects.npz``). Unlike the full model it
   fits the estimation noise freely instead of fixing it from the Estimation task.
3. ``noise_per_cond_inf``: the pre-registered single-noise inference model fitted per incentive condition
   (``inf`` in ``fit_noise_per_cond.npz``), scored on the whole dataset by summing the three per-condition
   log-likelihoods (3 free parameters, no link between conditions; ``noise_per_cond_global_aicc_table``).

Participants with all three fits are pooled over Experiments 1 and 2 (n = 216). The full model is scored at its
best BADS (MAP) solution; the other two models at their own best fit.

Output (``results/exploratory/merged/modeling/stats/full_vs_inference_only/``):
``aicc_full_vs_infv_vs_noise_per_cond_by_participant.csv``. The tables stage writes
``groupbmc_full_vs_infv_vs_noise_per_cond.csv``.

Usage (from repo root):
    python scripts/exploratory/stats/run_full_vs_inference_only_bmc.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

from motinf.exploratory.single_modulation_fits import load_categ_single_incentive_fit
from motinf.exploratory.full_model_comparison import (
    full_model_fit_table,
    noise_per_cond_global_aicc_table,
)
from motinf.exploratory.table_stats import FULL_VS_INF_AICC

SEGMENTS = ["exp1", "exp2"]


def main() -> None:
    out_dir = ROOT / "results" / "exploratory" / "merged" / "modeling" / "stats" / "full_vs_inference_only"
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = []
    for seg in SEGMENTS:
        df_full = full_model_fit_table(ROOT, seg)[["participant", "aicc_bads_full"]].copy()
        df_full["participant"] = df_full["participant"].astype(str)

        fit = load_categ_single_incentive_fit(ROOT / "results" / "prereg" / seg / "model_fit" / "fit_inc_effects.npz")
        df_infv = pd.DataFrame({"participant": fit.participants, "aicc_inf_v_prereg": fit.aicc["inf_v"]})
        df_infv["participant"] = df_infv["participant"].astype(str)

        df_npc = noise_per_cond_global_aicc_table(ROOT, seg)[["participant", "aicc_noise_per_cond_inf"]].copy()
        df_npc["participant"] = df_npc["participant"].astype(str)

        merged = df_full.merge(df_infv, on="participant", how="inner").merge(df_npc, on="participant", how="inner")
        merged["segment"] = seg
        print(f"[{seg}] full n={len(df_full)}, inf_v_prereg n={len(df_infv)}, noise_per_cond_inf n={len(df_npc)}, "
              f"all three n={len(merged)}")
        chunks.append(merged)

    df_all = pd.concat(chunks, ignore_index=True).dropna(subset=list(FULL_VS_INF_AICC.values()))
    df_all.to_csv(out_dir / "aicc_full_vs_infv_vs_noise_per_cond_by_participant.csv", index=False)
    print(f"Saved: {out_dir / 'aicc_full_vs_infv_vs_noise_per_cond_by_participant.csv'} ({len(df_all)} participants)")


if __name__ == "__main__":
    main()
