"""Per-participant tables of the pre-registered fits (pooled sample).

Inputs (per segment of ``cohort_subexperiments(--exp-name)``): ``results/prereg/<seg>/model_fit/``
(``fit_noise_per_cond.npz``, ``fit_inc_effects.npz``), ``results/prereg/<seg>/behavior/`` and
``data/processed/<seg>/data_clean_inf.csv``.

Outputs (``results/exploratory/<exp>/modeling/stats/model_fit_stats/params/``):

- ``aicc_noise_per_cond_by_participant.csv``, ``aicc_inc_effects_by_participant.csv``: per-participant AICc of the
  single-noise models in each condition and of the single-modulation models, pooled sample;
- ``pred_choice_var_single_modulation.csv``: choice variability predicted by each single-modulation model
  per participant, incentive and sequence length; ``var_red_by_seqlen_single_modulation.csv``: the reduction
  under incentives per sequence length, data and each model (mean, s.e.m.; Supp. Figure 4d).

The group BMC (Figure 2b, Supp. Figure 4a), the error of the predicted incentive effect (Supp. Figure 4c) and the
correlation of the incentive effects between tasks (Results) are computed from these tables by the ``tables`` stage
(``motinf.exploratory.table_stats``).

Usage (from the repository root)::

    python scripts/exploratory/stats/run_inference_model_stats.py --exp-name merged
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

from motinf.exploratory.choice_var_seqlen_preprocess import merge_and_preprocess_choice_var_beh_pred
from motinf.exploratory.cohorts import (
    cohort_subexperiments,
    merge_behavior_seqlen,
    merge_inf_trials_for_choice_var,
    participant_prefixed,
    validate_prereg_inputs_exist,
)
from motinf.exploratory.single_modulation_fits import (
    SINGLE_MODULATION_MODELS,
    CategSingleIncentiveFit,
    load_categ_single_incentive_fit_merged,
    pooled_reduction_summary_by_incentive_abs,
    predict_choicevar_per_seqlen_categ,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model-fit statistics of the pre-registered fits (pooled sample).")
    parser.add_argument(
        "--exp-name",
        default="merged",
        help="exp1, exp2 or merged (pools exp1 and exp2).",
    )
    return parser.parse_args()


def _single_modulation_predictions(
    fit: CategSingleIncentiveFit,
    params_dir: Path,
    df_trials: pd.DataFrame,
    df_beh_seq: pd.DataFrame,
) -> None:
    """Choice variability predicted by each single-modulation model, per participant x incentive x sequence length,
    from the pre-registered generative model (``single_modulation_fits.predict_choicevar_per_seqlen_categ``), and the
    reduction under incentives per sequence length, data and prediction (Supplementary Figure 4d).

    Writes ``pred_choice_var_single_modulation.csv`` and ``var_red_by_seqlen_single_modulation.csv``, so that the
    figure needs no trial data."""
    beh = df_beh_seq.copy()
    beh["participant"] = beh["participant"].astype(str)
    beh["incentive"] = beh["incentive"].astype(float)
    beh["seqlen"] = beh["seqlen"].astype(int)
    preds, summaries = [], []
    for m in SINGLE_MODULATION_MODELS:
        pred = predict_choicevar_per_seqlen_categ(df_trials, fit, m)
        preds.append(pred.assign(model=m))
        m_nat = merge_and_preprocess_choice_var_beh_pred(beh, pred)
        summaries.append(pooled_reduction_summary_by_incentive_abs(m_nat, m))
    for name, df in (("pred_choice_var_single_modulation.csv", pd.concat(preds, ignore_index=True)),
                     ("var_red_by_seqlen_single_modulation.csv", pd.concat(summaries, ignore_index=True))):
        df.to_csv(params_dir / name, index=False)
        print(f"Saved: {params_dir / name}")


def _noise_per_cond_aicc(root: Path, segments: list[str]) -> pd.DataFrame:
    """AICc of the single-noise models per participant and condition (``aicc_<model>`` of each segment's
    ``fit_noise_per_cond.npz``), segments stacked in order (ids prefixed ``seg__`` when there are several): one row per
    condition x model x participant, in that nesting order."""
    prefix_pids = len(segments) > 1
    fits = [np.load(root / "results" / "prereg" / seg / "model_fit" / "fit_noise_per_cond.npz", allow_pickle=True)
            for seg in segments]
    incentives = np.asarray(fits[0]["incentives"], dtype=float)
    names = [str(s["name"]) for s in fits[0]["categ_model_specs"]]
    for seg, d in zip(segments, fits):
        if not np.array_equal(np.asarray(d["incentives"], dtype=float), incentives):
            raise ValueError(f"fit_noise_per_cond.npz of {seg!r}: incentives differ from {segments[0]!r}.")
        if [str(s["name"]) for s in d["categ_model_specs"]] != names:
            raise ValueError(f"fit_noise_per_cond.npz of {seg!r}: models differ from {segments[0]!r}.")
    participants = [participant_prefixed(seg, str(p)) if prefix_pids else str(p)
                    for seg, d in zip(segments, fits) for p in d["participants"]]
    aicc = {m: np.vstack([np.asarray(d[f"aicc_{m}"], dtype=float) for d in fits]) for m in names}
    return pd.DataFrame([{"participant": p, "incentive": float(inc), "model": m, "aicc": float(aicc[m][pi, ji])}
                         for ji, inc in enumerate(incentives) for m in names for pi, p in enumerate(participants)])


def _write_aicc_tables(root: Path, cohort_segs: list[str], fit: CategSingleIncentiveFit, params_dir: Path) -> None:
    """Per-participant AICc of the single-noise models in each condition (``aicc_noise_per_cond_by_participant.csv``)
    and of the single-modulation models (``aicc_inc_effects_by_participant.csv``), pooled over the experiments, in the
    order of the fit files. The ``tables`` stage runs the group BMC on them (``table_stats``)."""
    _noise_per_cond_aicc(root, cohort_segs).to_csv(params_dir / "aicc_noise_per_cond_by_participant.csv", index=False)
    pd.DataFrame([{"participant": p, "model": m, "aicc": float(fit.aicc[m][pi])}
                  for m in fit.model_names for pi, p in enumerate(fit.participants)]).to_csv(
        params_dir / "aicc_inc_effects_by_participant.csv", index=False)


def main() -> None:
    args = parse_args()
    cohort_segs = cohort_subexperiments(args.exp_name)
    validate_prereg_inputs_exist(ROOT, cohort_segs)

    out_dir = ROOT / "results" / "exploratory" / args.exp_name / "modeling" / "stats" / "model_fit_stats"
    params_dir = out_dir / "params"
    for d in (params_dir, out_dir / "correlations"):
        d.mkdir(parents=True, exist_ok=True)

    # Single-modulation models: predicted choice variability over sequence length (Supp. Fig. 4c, d)
    fit = load_categ_single_incentive_fit_merged(ROOT, cohort_segs)
    _single_modulation_predictions(fit, params_dir, merge_inf_trials_for_choice_var(ROOT, cohort_segs),
                                   merge_behavior_seqlen(ROOT, cohort_segs))

    # Per-participant AICc of the pooled sample, for the group BMC of the tables stage (Figure 2b, Supp. Fig. 4a);
    # per experiment, the group BMC comes from scripts/prereg/run_model_stats.py
    if len(cohort_segs) > 1:
        _write_aicc_tables(ROOT, cohort_segs, fit, params_dir)
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
