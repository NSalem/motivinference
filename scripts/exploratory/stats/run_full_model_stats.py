"""Per-participant tables of the full model (Figure 3, Supplementary Figure 6, Supplementary Table 4).

Inputs: ``results/exploratory/<segment>/modeling/fit_files/fit_full_model.npz``
(``scripts/exploratory/fits/fit_full_model.py``), ``data/processed/<segment>/data_clean_inf.csv``
and the behaviour summaries in ``results/prereg/<segment>/behavior/``; ``--exp-name merged`` pools
exp1 and exp2 (``motinf.exploratory.cohorts``).

Outputs (``results/exploratory/<exp-name>/modeling/stats/full_model_stats/``):

- ``params/params_wide.csv``: fitted parameters plus effective noise and predicted variance per
  noise source with and without incentives.
- ``knockout/``: predicted choice variability of the full model and of the knockouts without each
  incentive modulation, and the observed / predicted incentive effect per sequence length (Fig. 3c).
- ``correlations/param_behavior_{main,diff}.csv``: choice-variance midpoint and slope (data and full-model
  predictions) against the parameters (Supp. Fig. 6); same participants as the knockout analyses
  (choice-variability exclusion, 5 robust SDs).

The tests on these tables (δ against 0, variance partition, correlation of the δ, robust regressions of Supp.
Table 4, prediction error of the knockouts) are computed by the ``tables`` stage (``motinf.exploratory.table_stats``).

Usage (from repo root)::

    python scripts/exploratory/stats/run_full_model_stats.py --exp-name merged
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

from motinf.exploratory.choice_var_seqlen_preprocess import (
    merge_and_preprocess_choice_var_beh_pred,
    var_red_natural_long_by_seqlen,
)
from motinf.exploratory.cohorts import (
    cohort_subexperiments,
    merge_behavior_seqlen,
    merge_data_clean_inf,
    validate_full_model_inputs_exist,
)
from motinf.exploratory.full_model_results import (
    CHANNELS,
    add_effective_and_variance,
    aggregate_predictions_per_seqlen,
    compute_model_predictions,
    knockout_model_specs,
    load_full_model_params_for_segments,
)
from motinf.exploratory.stats import fit_slopes_intercepts


# ---------------------------------------------------------------------------
# Argument parsing / IO helpers
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run model-result statistics for the fixed-ang fit."
    )
    parser.add_argument(
        "--exp-name",
        default="merged",
        help="exp1, exp2 or merged (pools exp1 and exp2; see motinf.exploratory.cohorts).",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help=(
            "Override output directory (default: "
            "results/exploratory/<exp>/modeling/stats/full_model_stats)."
        ),
    )
    return parser.parse_args()


def knockout_tables(
    df_pars: pd.DataFrame,
    df_inf: pd.DataFrame,
    cohort_segs: list[str],
    df_pred_full_tab: pd.DataFrame,
    out_dir: Path,
) -> list[str]:
    """Full model vs knockouts (one incentive modulation set to zero).

    Writes ``knockout/knockout_predictions.csv`` (predicted choice variability per participant,
    incentive and sequence length; Figure 3b-d) and ``knockout/knockout_var_red_natural_by_seqlen_long.csv``
    (observed and predicted incentive effect per sequence length; Figure 3c). Data and predictions go
    through the same choice-variability preprocessing (``merge_and_preprocess_choice_var_beh_pred``:
    participants with extreme values excluded). The per-participant fit of the incentive effect
    (Figure 3d) is computed from these tables by ``table_stats.knockout_prediction_error``.

    Returns the participants kept by that exclusion (more than 5 robust SDs from the median in
    any incentive x sequence length cell), which :func:`param_behavior_tables` reuses.
    """
    specs = knockout_model_specs()
    df_data_var = merge_behavior_seqlen(ROOT, cohort_segs)
    df_data_var["participant"] = df_data_var["participant"].astype(str)
    df_data_var["incentive"] = df_data_var["incentive"].astype(float)
    df_data_var["seqlen"] = df_data_var["seqlen"].astype(int)
    df_data_var = df_data_var[["participant", "incentive", "seqlen", "choice_var", "choice_var_log"]]

    df_beh_nat_filt = merge_and_preprocess_choice_var_beh_pred(
        df_data_var, df_pred_full_tab, report_label="full_model_stats knockout (data natural filter)"
    )
    pids_keep = sorted(df_beh_nat_filt["participant"].astype(str).unique())

    pred_chunks: list[pd.DataFrame] = []
    for spec in specs:
        agg = aggregate_predictions_per_seqlen(
            compute_model_predictions(df_inf, df_pars, knockout_motivs=spec["knockouts"]))
        pred_chunks.append(agg.assign(model_id=spec["id"]))

    df_predictions = pd.concat(pred_chunks, ignore_index=True)
    df_predictions.to_csv(out_dir / "knockout_predictions.csv", index=False)

    # Incentive variance reduction vs seqlen (natural scale, raw V0−V1), filtered participant set.
    long_data = var_red_natural_long_by_seqlen(df_beh_nat_filt, "choice_var").assign(model_id="data")
    var_red_long_chunks = [long_data]
    for mid in (s["id"] for s in specs):
        sub = df_predictions[df_predictions["model_id"] == mid]
        sub = sub[sub["participant"].astype(str).isin(pids_keep)]
        var_red_long_chunks.append(var_red_natural_long_by_seqlen(sub, "pred_choice_var").assign(model_id=mid))
    pd.concat(var_red_long_chunks, ignore_index=True).to_csv(
        out_dir / "knockout_var_red_natural_by_seqlen_long.csv", index=False
    )
    return pids_keep


def _build_param_behavior_table(
    df_pars: pd.DataFrame,
    cohort_segs: list[str],
    df_pred_full: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build per-participant tables.

    Returns
    -------
    df_main : per participant, neutral condition only (incentive 0, where the full model's
              predictions depend on the noise parameters alone), with
              `sd_X`, choice-variance intercept and slope (data and model).
    df_diff : per participant `motiv_X`, `intercept_diff`, `slope_diff`,
              `intercept_mod_diff`, `slope_mod_diff` (|incentive| = 1 minus 0).
    """
    df_seqlen = merge_behavior_seqlen(ROOT, cohort_segs)
    df_seqlen["participant"] = df_seqlen["participant"].astype(str)
    df_seqlen["incentive"] = df_seqlen["incentive"].astype(float)
    df_seqlen["seqlen"] = df_seqlen["seqlen"].astype(int)

    # Intercepts/slopes for full-model predictions per (participant, incentive_abs).
    df_pred_for_slopes = df_pred_full.copy()
    df_pred_for_slopes["incentive_abs"] = (
        np.abs(df_pred_for_slopes["incentive"].astype(float)).round().astype(int)
    )
    df_slopes_mod_abs = fit_slopes_intercepts(
        df_pred_for_slopes,
        dv="choicevar",
        iv="seqlen",
        group_vars=["participant", "incentive_abs"],
    ).rename(
        columns={
            "incentive": "incentive_abs",
            "intercept": "intercept_mod",
            "slope": "slope_mod",
        }
    )

    df_slopes_mod_abs["participant"] = df_slopes_mod_abs["participant"].astype(str)
    df_slopes_mod_abs["incentive_abs"] = df_slopes_mod_abs["incentive_abs"].astype(int)

    # Data intercepts/slopes on choice_var, per participant × |incentive|.
    df_seqlen["incentive_abs"] = df_seqlen["incentive"].abs().round().astype(int)
    df_slopes_abs = fit_slopes_intercepts(
        df_seqlen,
        dv="choice_var",
        iv="seqlen",
        group_vars=["participant", "incentive_abs"],
    ).rename(columns={"incentive": "incentive_abs"})
    df_slopes_abs = df_slopes_abs[
        ["participant", "incentive_abs", "intercept", "slope"]
    ].copy()
    df_slopes_mod_abs = df_slopes_mod_abs[
        ["participant", "incentive_abs", "intercept_mod", "slope_mod"]
    ].copy()

    df_abs = df_pars[["participant"] + [f"sd_{ch}" for ch in CHANNELS]].copy()
    df_abs = df_abs.merge(df_slopes_abs, on="participant", how="inner")
    df_abs = df_abs.merge(df_slopes_mod_abs, on=["participant", "incentive_abs"], how="left")

    # df_main: neutral condition only, so baseline noise parameters are related to baseline behaviour
    # (incentive effects, which also depend on the modulations, are df_diff).
    df_main = (
        df_abs[df_abs["incentive_abs"] == 0].groupby("participant", as_index=False)[
            [f"sd_{ch}" for ch in CHANNELS]
            + ["intercept", "slope", "intercept_mod", "slope_mod"]
        ]
        .mean()
    )

    # df_diff: incentive=1 minus incentive=0 per metric.
    df_diff_rows: list[dict] = []
    for pid, sub in df_abs.groupby("participant", observed=True):
        wide = sub.set_index("incentive_abs")
        row: dict = {"participant": str(pid)}
        # Use direct motiv parameters for modulation analyses.
        p0 = df_pars[df_pars["participant"] == str(pid)]
        if not p0.empty:
            for ch in CHANNELS:
                row[f"motiv_{ch}"] = float(p0[f"motiv_{ch}"].iloc[0])
        for col in ("intercept", "slope", "intercept_mod", "slope_mod"):
            try:
                row[f"{col}_diff"] = float(wide.loc[1, col]) - float(wide.loc[0, col])
            except KeyError:
                row[f"{col}_diff"] = np.nan
        df_diff_rows.append(row)
    df_diff = pd.DataFrame(df_diff_rows)

    return df_main, df_diff


def param_behavior_tables(
    df_pars: pd.DataFrame,
    cohort_segs: list[str],
    out_dir: Path,
    df_pred_full: pd.DataFrame,
    *,
    keep_participants: list[str],
) -> None:
    """Choice-variance midpoint and slope (data and full-model predictions) with the noise parameters
    (``main`` block, baseline) and with the incentive effects and modulations (``diff`` block), per
    participant (``param_behavior_{main,diff}.csv``; Supp. Figure 6). The robust regressions of Supp.
    Table 4 are fitted to these tables by ``table_stats.param_behavior_rlm``.

    The tables use ``keep_participants``: the participants kept by the choice-variability exclusion
    of the other sequence-length analyses (:func:`knockout_tables`; Figs 2d, 3c,d), so midpoint and slope are only
    computed from participants without extreme cells."""
    df_main, df_diff = _build_param_behavior_table(df_pars, cohort_segs, df_pred_full)
    keep = set(map(str, keep_participants))
    df_main = df_main[df_main["participant"].astype(str).isin(keep)].reset_index(drop=True)
    df_diff = df_diff[df_diff["participant"].astype(str).isin(keep)].reset_index(drop=True)
    print(f"choice-variability exclusion: N = {len(df_main)} (main), {len(df_diff)} (diff) "
          f"of {df_pars['participant'].nunique()}")
    df_main.to_csv(out_dir / "param_behavior_main.csv", index=False)
    df_diff.to_csv(out_dir / "param_behavior_diff.csv", index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    exp_name = args.exp_name

    cohort_segs = cohort_subexperiments(exp_name)
    validate_full_model_inputs_exist(ROOT, cohort_segs)

    if args.out_dir.strip():
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = (
            ROOT
            / "results"
            / "exploratory"
            / exp_name
            / "modeling"
            / "stats"
            / "full_model_stats"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    params_dir = out_dir / "params"
    knockout_dir = out_dir / "knockout"
    correlations_dir = out_dir / "correlations"
    for d in (params_dir, knockout_dir, correlations_dir):
        d.mkdir(parents=True, exist_ok=True)

    df_pars_raw = load_full_model_params_for_segments(ROOT, cohort_segs)
    df_inf = merge_data_clean_inf(ROOT, cohort_segs)
    df_inf["seqlen"] = df_inf["seqlen"].astype(int)
    df_inf["incentive"] = df_inf["incentive"].astype(float)
    mean_seqlen = float(df_inf["seqlen"].mean())
    df_pars = add_effective_and_variance(df_pars_raw, mean_seqlen)
    df_pars.to_csv(params_dir / "params_wide.csv", index=False)

    print(f"____STATS full model | exp={exp_name} | n={len(df_pars)} | seqlen_mean={mean_seqlen:.3f}____")

    df_pred_full = compute_model_predictions(df_inf, df_pars, knockout_motivs=())

    print("Knockouts")
    pids_choice_var = knockout_tables(
        df_pars, df_inf, cohort_segs, aggregate_predictions_per_seqlen(df_pred_full), knockout_dir
    )

    print("Parameter–behaviour tables")
    param_behavior_tables(
        df_pars,
        cohort_segs,
        correlations_dir,
        df_pred_full,
        keep_participants=pids_choice_var,
    )
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
