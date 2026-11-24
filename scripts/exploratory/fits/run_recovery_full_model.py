# -*- coding: utf-8 -*-
"""Parameter recovery for the full model (Supplementary Figure 5).

Per participant (with their real trial sequences) and replicate:

1. generative parameters are drawn from the full model's priors;
2. Sequence Categorization choices and Stimulus Reproduction responses are simulated;
3. ``sd_ang`` and ``motiv_ang`` are derived from the simulated reproduction errors, as in the real fits;
4. the free parameters are fitted to the simulated choices (BADS, then VBMC unless ``--no-vbmc-after-bads``).

``--exp-name merged`` pools Experiments 1 and 2 (participant ids prefixed ``exp1__`` / ``exp2__``). Seeds depend on a
participant's index within their own experiment, so a merged run equals the exp1 and exp2 runs combined.

Input: ``data/processed/<exp>/data_clean_{est,inf}.csv``.
Outputs (``results/exploratory/<exp>/modeling/param_recovery_full_model/``): ``parameter_recovery_runs.csv`` (one
row per fit, ``<param>_true`` / ``<param>_fit``; read by ``sfig5.py``), ``parameter_recovery_long.csv`` and
``parameter_recovery_summary.json`` (settings and recovery statistics).

Usage (from the repository root), with the paper's settings (one simulated dataset per participant, seed 42):
    python scripts/exploratory/fits/run_recovery_full_model.py --exp-name merged --bads-nit 10 --vbmc-after-bads
"""

from __future__ import annotations

import argparse
import json
import random as _random
from pathlib import Path

import numpy as np
import pandas as pd

from motinf.exploratory.cohorts import (
    apply_row_limits,
    cohort_subexperiments,
    merge_data_clean_est,
    merge_data_clean_inf,
)
from motinf.exploratory.full_model_fitting import (
    FIT_FREE_PARNAMES,
    GEN_PARNAMES,
    correlation,
    fit_full_model_participant,
    fixed_ang_columns,
    get_fixed_ang_priors,
    sample_theta_from_priors,
    simulate_estim_resp_joint,
    theta_to_params,
)
from motinf.exploratory.model import create_categ_model, simulate_choice, subset_priors

ROOT = Path(__file__).resolve().parents[3]
GLOBAL_SEED = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Parameter recovery for fixed-angular variant: derive sd_ang/motiv_ang from "
            "simulated estimation errors, then fit categorization free parameters."
        )
    )
    p.add_argument("--exp-name", default="merged")
    p.add_argument("--n-reps", type=int, default=1)
    p.add_argument("--seed", type=int, default=GLOBAL_SEED)
    p.add_argument("--bads-nit", type=int, default=10)
    p.add_argument(
        "--vbmc-after-bads",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After multi-start BADS, run VBMC from the best BADS parameters; recovery metrics use the VBMC "
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
        help="Output directory override; defaults to results/exploratory/<exp>/modeling/param_recovery_full_model.",
    )
    p.add_argument(
        "--sanity-run",
        action="store_true",
        help="Quick mode: n-reps=1, max-nsubs=2, bads-nit=1.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.sanity_run:
        args.n_reps = 1
        args.max_nsubs = 2
        args.bads_nit = 1

    _random.seed(args.seed)
    np.random.seed(args.seed)

    segments = cohort_subexperiments(args.exp_name)
    exp_name = "merged" if len(segments) > 1 else segments[0]

    # Merged cohorts: participants become "<seg>__<id>" (exp1 first, then exp2).
    df_inf = merge_data_clean_inf(ROOT, segments)
    df_inf = apply_row_limits(df_inf, args.max_ntrials, args.max_nsess)
    df_est = merge_data_clean_est(ROOT, segments)
    df_est = apply_row_limits(df_est, args.max_ntrials, args.max_nsess)

    part_inf = np.asarray(pd.unique(df_inf["participant"]), dtype=object)
    part_est = np.asarray(pd.unique(df_est["participant"]), dtype=object)
    participants = np.array([p for p in part_inf if p in set(part_est)], dtype=object)
    if args.max_nsubs > 0:
        participants = participants[: args.max_nsubs]
    if len(participants) == 0:
        raise ValueError("No participants available after filtering.")

    if args.out_dir.strip():
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = (
            ROOT
            / "results"
            / "exploratory"
            / exp_name
            / "modeling"
            / "param_recovery_full_model"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    prior_full = get_fixed_ang_priors()
    gen_priors = subset_priors(prior_full, GEN_PARNAMES)
    fit_priors = subset_priors(prior_full, FIT_FREE_PARNAMES) if args.use_priors else {}
    missing_priors = [p for p in GEN_PARNAMES if p not in gen_priors]
    if missing_priors:
        raise KeyError(f"Missing priors for generative parameters: {missing_priors}")

    model_fit = create_categ_model(FIT_FREE_PARNAMES)
    # Seeds use each participant's index *within its experiment*, so a merged run
    # reproduces the single-experiment runs participant for participant.
    seg_ip: dict[str, int] = {}
    seg_rep_idx: dict[str, int] = {}
    runs: list[dict] = []
    long_rows: list[dict] = []
    n_fail = 0

    print(f"____MODEL fixed_ang_from_estim | out_dir={out_dir}____")

    for ip, sub in enumerate(participants):
        seg = str(sub).split("__", 1)[0] if len(segments) > 1 else segments[0]
        ip_seg = seg_ip.get(seg, 0)
        seg_ip[seg] = ip_seg + 1
        df_c = df_inf[df_inf["participant"] == sub].reset_index(drop=True)
        df_e = df_est[df_est["participant"] == sub].reset_index(drop=True)
        if len(df_c) == 0 or len(df_e) == 0:
            continue

        for rep in range(args.n_reps):
            rep_idx = seg_rep_idx.get(seg, 0)
            seg_rep_idx[seg] = rep_idx + 1
            rng = np.random.default_rng(args.seed + rep_idx * 1_000_003 + ip_seg * 10_007)

            theta_true = sample_theta_from_priors(GEN_PARNAMES, gen_priors, rng)
            params_true = theta_to_params(GEN_PARNAMES, theta_true)

            np.random.seed(int(rng.integers(0, 2**31 - 1)))
            choice_sim, _, _, _ = simulate_choice(params_true, df_c)
            df_c_sim = df_c.copy()
            df_c_sim["choice"] = choice_sim
            df_e_sim = df_e.copy()
            df_e_sim["resp_estim"] = simulate_estim_resp_joint(params_true, df_e, rng)

            print(
                f"FIT fixed-ang | sub {ip + 1}/{len(participants)} ({sub}), "
                f"rep {rep + 1}/{args.n_reps} | bads_nit={args.bads_nit}"
                f"{' | vbmc-after-bads' if args.vbmc_after_bads else ''}"
            )
            try:
                res = fit_full_model_participant(
                    df_c_sim, df_e_sim, model_fit, fit_priors, args.bads_nit, vbmc=bool(args.vbmc_after_bads)
                )
                fixed = res["fixed"]
                theta_fit = np.asarray(res["params"], dtype=float).ravel()
                ll, aicc, bic = float(res["ll"]), float(res["aicc"]), float(res["bic"])
                ll_bads, aicc_bads, bic_bads = float(res["bads_ll"]), float(res["bads_aicc"]), float(res["bads_bic"])
                elbo_vbmc = float(res["elbo"])
                ok = np.all(np.isfinite(theta_fit)) and np.isfinite(ll)
                err_msg = ""
            except Exception as exc:
                fixed = None
                theta_fit = np.full(len(FIT_FREE_PARNAMES), np.nan)
                ll = aicc = bic = ll_bads = aicc_bads = bic_bads = elbo_vbmc = np.nan
                ok = False
                err_msg = str(exc)
                n_fail += 1

            row = {
                "participant": str(sub),
                "rep": rep,
                "model_name": "fixed_ang_from_estim",
                "n_trials_categ": int(len(df_c_sim)),
                "n_trials_estim": int(len(df_e_sim)),
                "n_trials_total": int(len(df_c_sim) + len(df_e_sim)),
                "ok": bool(ok),
                "error": err_msg,
                "ll": ll,
                "aicc": aicc,
                "bic": bic,
                "ll_bads": ll_bads,
                "aicc_bads": aicc_bads,
                "bic_bads": bic_bads,
                "elbo_vbmc": elbo_vbmc,
                **fixed_ang_columns(fixed),
                "sd_ang_true": float(params_true["sd_ang"]),
                "motiv_ang_true": float(params_true["motiv_ang"]),
            }
            for i, pn in enumerate(FIT_FREE_PARNAMES):
                true_val = float(params_true[pn])
                fit_val = float(theta_fit[i]) if i < len(theta_fit) else np.nan
                row[f"{pn}_true"] = true_val
                row[f"{pn}_fit"] = fit_val
                row[f"{pn}_bias"] = float(fit_val - true_val) if ok and np.isfinite(fit_val) else np.nan
                long_rows.append(
                    {
                        "participant": str(sub),
                        "rep": rep,
                        "model_name": "fixed_ang_from_estim",
                        "param": pn,
                        "true": true_val,
                        "fit": fit_val,
                        "bias": float(fit_val - true_val) if ok and np.isfinite(fit_val) else np.nan,
                        "ok": bool(ok),
                    }
                )
            runs.append(row)

    if not runs:
        raise RuntimeError("No recovery rows produced.")

    df_runs = pd.DataFrame(runs)
    df_long = pd.DataFrame(long_rows)
    out_runs = out_dir / "parameter_recovery_runs.csv"
    out_long = out_dir / "parameter_recovery_long.csv"
    df_runs.to_csv(out_runs, index=False)
    df_long.to_csv(out_long, index=False)

    ok_long = df_long[df_long["ok"]].copy()
    per_param: dict[str, dict] = {}
    for pn in FIT_FREE_PARNAMES:
        d = ok_long[ok_long["param"] == pn]
        if d.empty:
            per_param[pn] = {
                "n": 0,
                "mae": float("nan"),
                "rmse": float("nan"),
                "bias_mean": float("nan"),
                "pearson_r": float("nan"),
                "spearman_r": float("nan"),
            }
            continue
        true_v = d["true"].to_numpy(dtype=float)
        fit_v = d["fit"].to_numpy(dtype=float)
        bias_v = d["bias"].to_numpy(dtype=float)
        per_param[pn] = {
            "n": int(len(d)),
            "mae": float(np.nanmean(np.abs(bias_v))),
            "rmse": float(np.sqrt(np.nanmean((bias_v) ** 2))),
            "bias_mean": float(np.nanmean(bias_v)),
            "pearson_r": correlation(true_v, fit_v, method="pearson"),
            "spearman_r": correlation(true_v, fit_v, method="spearman"),
        }

    summary = {
        "model_name": "fixed_ang_from_estim",
        "generative_parnames": GEN_PARNAMES,
        "fit_free_parnames": FIT_FREE_PARNAMES,
        "fixed_from_estim": ["sd_ang", "motiv_ang"],
        "exp_name": exp_name,
        "seed": int(args.seed),
        "n_reps": int(args.n_reps),
        "bads_nit": int(args.bads_nit),
        "vbmc_after_bads": bool(args.vbmc_after_bads),
        "max_nsubs": int(args.max_nsubs),
        "max_ntrials": int(args.max_ntrials),
        "max_nsess": int(args.max_nsess),
        "use_priors_fit": bool(args.use_priors),
        "n_rows": int(len(df_runs)),
        "n_ok": int(df_runs["ok"].sum()),
        "n_fail": int(n_fail),
        "per_param": per_param,
    }

    summary_path = out_dir / "parameter_recovery_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved: {out_runs}")
    print(f"Saved: {out_long}")
    print(f"Saved: {summary_path}")
    print(f"fixed_ang_from_estim: n_ok={summary['n_ok']} / n_rows={summary['n_rows']}")


if __name__ == "__main__":
    main()
