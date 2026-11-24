"""Statistics computed from participant-level tables only (the first step of the ``tables`` stage).

Reads the tables the ``stats`` stage wrote to ``results/`` and recomputes, without trial data or fit files
(``motinf.exploratory.table_stats``):

- per experiment and pooled (``behavior/stats/behav_extra/params``): natural-scale choice-variability rmANOVA,
  R² of the inference-only model, accuracy against the ideal observer, trials removed by the response-time cut;
- pooled (``modeling/stats/full_model_stats``): full-model δ against 0, variance partition, correlation of the δ,
  robust regressions of Supplementary Table 4, prediction error of the full model and of the knockouts;
- pooled (``modeling/stats/model_fit_stats``): correlation of the incentive effects between tasks, group Bayesian
  model selection of the single-noise and single-modulation models, prediction error of the single-modulation
  models;
- pooled (``modeling/stats/full_vs_inference_only``): group Bayesian model selection of the full model against
  the inference-only models.

Usage (from the repository root)::

    python scripts/exploratory/stats/run_table_stats.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

from motinf.exploratory.table_stats import run_table_stats

if __name__ == "__main__":
    run_table_stats(ROOT)
    print("Table statistics written under results/")
