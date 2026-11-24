# -*- coding: utf-8 -*-
"""Fit the full model to each participant (Figure 3, Supplementary Figure 6, Supplementary Tables 4-5).

The full model has the three sources of noise (sensory ``sd_ang``, inference ``sd_inf``, selection ``sd_sel``) and an
incentive effect on each (``motiv_*`` = δ = log σ£0 − log σ±£1). Per participant:

1. ``sd_ang`` and ``motiv_ang`` are not fitted to the choices: they are fixed from the participant's Stimulus
   Reproduction errors (``derive_fixed_ang_from_estim_errors``).
2. The four free parameters (``sd_inf``, ``sd_sel``, ``motiv_inf``, ``motiv_sel``) are fitted to the Sequence
   Categorization choices with BADS from ``--bads-nit`` random starting points, with the priors of Table 4.
3. VBMC starts from the best BADS solution (default; ``--no-vbmc-after-bads`` stops after BADS). The paper reports
   the VBMC posterior means.

Inputs: ``data/processed/<exp>/data_clean_{est,inf}.csv``.
Outputs: ``results/exploratory/<exp>/modeling/fit_files/fit_full_model.npz`` (BADS and VBMC estimates per
participant) and ``results/exploratory/<exp>/modeling/full_model_fit/{fit_runs.csv,fit_summary.json}``.
A run fits every participant (``--max-nsubs`` limits it to the first ones, for a quick run) and replaces these files.

Usage (from the repository root), with the paper's settings (seed 42, 10 BADS starts), once per experiment:
    python scripts/exploratory/fits/fit_full_model.py --exp-name exp1
"""

from __future__ import annotations

import argparse
import json
import random as _random
from pathlib import Path

import numpy as np
import pandas as pd

from motinf.exploratory.full_model_cohort import (
    allocate_slots,
    fit_file_dict,
    load_clean_dataframes,
    ordered_cohort_participants,
)
from motinf.exploratory.full_model_fitting import (
    FIT_FREE_PARNAMES,
    fit_full_model_participant,
    fixed_ang_columns,
    get_fixed_ang_priors,
)
from motinf.exploratory.model import create_categ_model, subset_priors

ROOT = Path(__file__).resolve().parents[3]
GLOBAL_SEED = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Fit real data with fixed sd_ang/motiv_ang derived from estimation errors, "
            "then fit free categorization parameters."
        )
    )
    p.add_argument("--exp-name", default="exp1")
    p.add_argument("--seed", type=int, default=GLOBAL_SEED)
    p.add_argument("--bads-nit", type=int, default=10)
    p.add_argument(
        "--vbmc-after-bads",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After multi-start BADS, run VBMC from the best BADS parameters; primary metrics use the VBMC "
            "posterior mean. --no-vbmc-after-bads: BADS only, metrics from the best BADS parameters."
        ),
    )
    p.add_argument("--max-nsubs", type=int, default=0, help="0 = all participants.")
    p.add_argument("--max-ntrials", type=int, default=0, help="0 = all trials.")
    p.add_argument("--max-nsess", type=int, default=0, help="0 = all sessions.")
    p.add_argument(
        "--use-priors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the priors of Table 4 in the categorization fitting objective.",
    )
    p.add_argument(
        "--out-dir",
        default="",
        help=(
            "Output directory override for CSV outputs "
            "(default: results/exploratory/<exp>/modeling/full_model_fit). "
            "NPZ defaults to results/exploratory/<exp>/modeling/fit_files/fit_full_model.npz."
        ),
    )
    p.add_argument(
        "--sanity-run",
        action="store_true",
        help="Quick mode: max-nsubs=2, bads-nit=1.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.sanity_run:
        args.max_nsubs = 2
        args.bads_nit = 1

    _random.seed(args.seed)
    np.random.seed(args.seed)

    exp_name = args.exp_name
    df_inf, df_est = load_clean_dataframes(ROOT, exp_name, args.max_ntrials, args.max_nsess)
    cohort = ordered_cohort_participants(df_inf, df_est, args.max_nsubs)
    cohort_n = len(cohort)

    if args.out_dir.strip():
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = ROOT / "results" / "exploratory" / exp_name / "modeling" / "full_model_fit"
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_dir = ROOT / "results" / "exploratory" / exp_name / "modeling" / "fit_files"
    npz_dir.mkdir(parents=True, exist_ok=True)
    npz_path = npz_dir / "fit_full_model.npz"

    prior_full = get_fixed_ang_priors()
    fit_priors = subset_priors(prior_full, FIT_FREE_PARNAMES) if args.use_priors else {}
    model_fit = create_categ_model(FIT_FREE_PARNAMES)

    slots = allocate_slots(cohort_n)

    runs: list[dict] = []
    n_fail = 0

    print(f"____FIT fixed_ang_from_estim_real | cohort_n={cohort_n} | out_dir={out_dir}____")

    for i, sub in enumerate(cohort):
        df_c = df_inf[df_inf["participant"] == sub].reset_index(drop=True)
        df_e = df_est[df_est["participant"] == sub].reset_index(drop=True)
        if len(df_c) == 0 or len(df_e) == 0:
            continue

        print(
            f"FIT fixed-ang real | {i + 1}/{cohort_n} ({sub}) | bads_nit={args.bads_nit}"
            f"{' | vbmc-after-bads' if args.vbmc_after_bads else ''}"
        )
        try:
            res = fit_full_model_participant(
                df_c, df_e, model_fit, fit_priors, args.bads_nit, vbmc=bool(args.vbmc_after_bads)
            )
            fixed = res["fixed"]
            slots["pars_bads"][i] = np.asarray(res["bads_params"], dtype=float).ravel()
            slots["ll_bads"][i] = float(res["bads_ll"])
            slots["aicc_bads"][i] = float(res["bads_aicc"])
            slots["bic_bads"][i] = float(res["bads_bic"])
            slots["pars_vbmc"][i] = np.asarray(res["vbmc_params"], dtype=float).ravel()
            slots["pars_vbmc_sd"][i] = np.asarray(res["vbmc_params_sd"], dtype=float).ravel()
            slots["ll_vbmc_mean"][i] = float(res["vbmc_ll"])
            slots["aicc_vbmc_mean"][i] = float(res["vbmc_aicc"])
            slots["bic_vbmc_mean"][i] = float(res["vbmc_bic"])
            slots["elbo_vbmc"][i] = float(res["elbo"])
            slots["fixed_sd_ang"][i] = float(fixed["sd_ang_categ"])
            slots["fixed_motiv_ang"][i] = float(fixed["motiv_ang_categ"])
            params_for_row = np.asarray(res["params"], dtype=float).ravel()
            ll_row, aicc_row, bic_row = float(res["ll"]), float(res["aicc"]), float(res["bic"])
            ok = np.all(np.isfinite(params_for_row)) and np.isfinite(ll_row)
            err_msg = ""
        except Exception as exc:
            fixed = None
            slots["pars_bads"][i] = np.full(len(FIT_FREE_PARNAMES), np.nan)
            slots["pars_vbmc"][i] = np.array([])
            slots["pars_vbmc_sd"][i] = np.array([])
            params_for_row = np.full(len(FIT_FREE_PARNAMES), np.nan)
            ll_row = aicc_row = bic_row = np.nan
            ok = False
            err_msg = str(exc)
            n_fail += 1

        row = {
            "participant": str(sub),
            "model_name": "fixed_ang_from_estim_real",
            "n_trials_categ": int(len(df_c)),
            "n_trials_estim": int(len(df_e)),
            "n_trials_total": int(len(df_c) + len(df_e)),
            "ok": bool(ok),
            "error": err_msg,
            "ll": float(ll_row),
            "aicc": float(aicc_row),
            "bic": float(bic_row),
            "ll_bads": float(slots["ll_bads"][i]),
            "aicc_bads": float(slots["aicc_bads"][i]),
            "bic_bads": float(slots["bic_bads"][i]),
            "elbo_vbmc": float(slots["elbo_vbmc"][i]),
            **fixed_ang_columns(fixed),
        }
        for j, pn in enumerate(FIT_FREE_PARNAMES):
            row[f"{pn}_fit"] = float(params_for_row[j]) if j < len(params_for_row) else np.nan
        runs.append(row)

    if not runs:
        raise RuntimeError("No participant has trials in both tasks.")

    df_runs = pd.DataFrame(runs)
    np.savez(
        str(npz_path),
        **fit_file_dict(
            slots,
            cohort_participants=cohort,
            vbmc_after_bads=bool(args.vbmc_after_bads),
            use_priors_fit=bool(args.use_priors),
            bads_nit=int(args.bads_nit),
            seed=int(args.seed),
            max_nsubs=int(args.max_nsubs),
            max_ntrials=int(args.max_ntrials),
            max_nsess=int(args.max_nsess),
        ),
    )
    df_runs.to_csv(out_dir / "fit_runs.csv", index=False)

    summary = {
        "model_name": "fixed_ang_from_estim_real",
        "fit_free_parnames": FIT_FREE_PARNAMES,
        "fixed_from_estim": ["sd_ang", "motiv_ang"],
        "exp_name": exp_name,
        "seed": int(args.seed),
        "bads_nit": int(args.bads_nit),
        "vbmc_after_bads": bool(args.vbmc_after_bads),
        "max_nsubs": int(args.max_nsubs),
        "max_ntrials": int(args.max_ntrials),
        "max_nsess": int(args.max_nsess),
        "cohort_n": int(cohort_n),
        "use_priors_fit": bool(args.use_priors),
        "n_rows_csv": int(len(df_runs)),
        "n_ok_csv": int(df_runs["ok"].sum()),
        "n_fail": int(n_fail),
    }
    summary_path = out_dir / "fit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved {npz_path} and {out_dir / 'fit_runs.csv'}: {summary['n_ok_csv']}/{summary['n_rows_csv']} fits ok")


if __name__ == "__main__":
    main()
