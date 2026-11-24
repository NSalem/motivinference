# -*- coding: utf-8 -*-
"""Model recovery for the single-noise models (Supplementary Figure 2a).

In each simulation, and for each generative model (sensory, inference or selection noise only):

1. choices are simulated on every participant's neutral-condition trials, with the parameter fitted to that
   participant at £0 by the pre-registered fit;
2. all three models are refitted to the simulated choices (BADS, ``--bads-nit`` starts);
3. group Bayesian model selection (GroupBMC) picks the recovered model (highest expected frequency).

Inputs: ``data/processed/<exp>/data_clean_inf.csv``, ``results/prereg/<exp>/model_fit/fit_noise_per_cond.npz``.
Outputs (``results/exploratory/<exp>/modeling/model_identifiability/``): ``confusion_matrix_percent.csv`` (read by
``sfig2.py``), ``confusion_matrix_counts.csv``, ``replicate_bmc_results.csv``, the protected exceedance
probabilities and ``run_summary.txt`` (settings).

Usage (from the repository root), with the paper's settings (seed 42, 5 BADS starts):
    python scripts/exploratory/fits/run_identifiability_single_noise.py --exp-name exp1 --n-reps 20
"""

from __future__ import annotations

import argparse
import random as _random
from pathlib import Path

import numpy as np
import pandas as pd

from motinf.exploratory.cohorts import apply_row_limits
from motinf.exploratory.identifiability import replicate_bmc_row, summary_tail, write_confusion_outputs
from motinf.exploratory.model import create_categ_model, run_fit_categ, simulate_choice, trials_ok

ROOT = Path(__file__).resolve().parents[3]
NEUTRAL = 0.0  # incentive value of the neutral condition
MODEL_SPECS = [
    {"name": "ang", "parnames": ["sd_ang"]},
    {"name": "inf", "parnames": ["sd_inf"]},
    {"name": "sel", "parnames": ["sd_sel"]},
]


def _rel(p) -> str:
    """A path as recorded in the run summary: relative to the repository when inside it."""
    p = Path(p).resolve()
    return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Confusion matrix of model identifiability via simulate-and-refit + GroupBMC."
    )
    p.add_argument("--exp-name", default="exp1")
    p.add_argument("--n-reps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bads-nit", type=int, default=5)
    p.add_argument("--max-nsubs", type=int, default=0, help="0 means all participants.")
    p.add_argument(
        "--max-ntrials",
        type=int,
        default=0,
        help="0 means all trials; otherwise keep rows with trialN <= max-ntrials.",
    )
    p.add_argument(
        "--max-nsess",
        type=int,
        default=0,
        help="0 means all sessions; otherwise keep rows with session <= max-nsess.",
    )
    p.add_argument(
        "--results-npz",
        default="",
        help="Optional path override for fitted npz. Default uses results/prereg/<exp>/model_fit/fit_noise_per_cond.npz",
    )
    p.add_argument(
        "--out-dir",
        default="",
        help="Override output directory. Default: results/exploratory/<exp>/modeling/model_identifiability",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    _random.seed(args.seed)
    np.random.seed(args.seed)

    exp_name = args.exp_name
    data_inf_path = ROOT / "data" / "processed" / exp_name / "data_clean_inf.csv"
    if args.results_npz.strip():
        fit_npz_path = Path(args.results_npz).expanduser().resolve()
    else:
        fit_npz_path = ROOT / "results" / "prereg" / exp_name / "model_fit" / "fit_noise_per_cond.npz"
    if args.out_dir.strip():
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = ROOT / "results" / "exploratory" / exp_name / "modeling" / "model_identifiability"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_inf_path.exists():
        raise FileNotFoundError(f"Missing data file: {data_inf_path}")
    if not fit_npz_path.exists():
        raise FileNotFoundError(f"Missing fit file: {fit_npz_path}")

    df_inf = pd.read_csv(data_inf_path)
    df_inf["seqlen"] = df_inf["seqlen"].astype(int)
    df_inf = apply_row_limits(df_inf, args.max_ntrials, args.max_nsess)
    fit = np.load(fit_npz_path, allow_pickle=True)

    participants_fit = np.array(fit["participants"])
    incentives_fit = np.asarray(fit["incentives"], dtype=float)
    df_inf = df_inf.copy()
    df_inf["participant_key"] = df_inf["participant"].astype(str)
    df_inf["incentive"] = pd.to_numeric(df_inf["incentive"], errors="coerce")

    model_names = [m["name"] for m in MODEL_SPECS]
    models = {m["name"]: create_categ_model(m["parnames"]) for m in MODEL_SPECS}
    participants = participants_fit[: args.max_nsubs] if args.max_nsubs > 0 else participants_fit
    neutral_j = int(np.flatnonzero(np.isclose(incentives_fit, NEUTRAL))[0])

    # Generative parameters: each participant's fit in the neutral condition (None if missing or not finite).
    gen_params_by_model: dict[str, list[np.ndarray | None]] = {}
    for gen_name in model_names:
        arr = fit[f"pars_{gen_name}"]  # (n_participants, n_incentives), object
        n_par = len(models[gen_name]["parnames"])
        per_part: list[np.ndarray | None] = []
        for ip in range(len(participants)):
            theta = arr[ip, neutral_j]
            theta = None if theta is None else np.asarray(theta, dtype=float).ravel()
            if theta is not None and (theta.size != n_par or np.any(~np.isfinite(theta))):
                theta = None
            per_part.append(theta)
        if all(t is None for t in per_part):
            raise ValueError(f"No usable fitted parameters for generative model {gen_name!r} in {fit_npz_path}.")
        gen_params_by_model[gen_name] = per_part

    rep_rows = []
    fit_fail_count = 0
    skip_participant_count = 0

    total = len(model_names) * args.n_reps
    done = 0
    for igen, gen_name in enumerate(model_names):
        gen_parnames = models[gen_name]["parnames"]
        print(f"\n=== Generative model: {gen_name} ({args.n_reps} replicates) ===")

        for rep in range(args.n_reps):
            done += 1
            print(f"\n[{done}/{total}] Start {gen_name} replicate {rep + 1}/{args.n_reps}")
            aicc = np.full((len(participants), len(model_names)), np.nan, dtype=float)
            rep_fit_fail_start = fit_fail_count
            rep_skip_start = skip_participant_count

            for ip, sub in enumerate(participants):
                theta_gen = gen_params_by_model[gen_name][ip]
                df_sub = df_inf[
                    (df_inf["participant_key"] == str(sub)) & np.isclose(df_inf["incentive"], NEUTRAL)
                ].reset_index(drop=True)
                if theta_gen is None or df_sub.empty:
                    skip_participant_count += 1
                    continue

                params_gen = {n: float(theta_gen[k]) for k, n in enumerate(gen_parnames)}
                np.random.seed(args.seed + rep * 100003 + ip * 1009)  # per replicate and participant
                try:
                    sim_choice, _, _, _ = simulate_choice(params_gen, df_sub)
                except Exception:
                    fit_fail_count += 1
                    continue

                df_sim = df_sub.copy()
                df_sim["choice"] = sim_choice

                for jm, rec_name in enumerate(model_names):
                    print(
                        f"Generative model [{igen + 1}/{len(model_names)}], "
                        f"replicate [{rep + 1}/{args.n_reps}], "
                        f"participant [{ip + 1}/{len(participants)}], "
                        f"candidate model [{jm + 1}/{len(model_names)}]"
                    )
                    m = models[rec_name]
                    if not trials_ok(len(df_sim), len(m["parnames"])):
                        continue
                    try:
                        res = run_fit_categ(df_sim, m, bads_nit=args.bads_nit)
                        if np.isfinite(res["aicc"]):
                            aicc[ip, jm] = float(res["aicc"])
                    except Exception:
                        fit_fail_count += 1

            row = replicate_bmc_row(gen_name, rep, aicc, model_names)
            rep_rows.append(row)
            outcome = f"recovered={row['recovered_model']}" if row["valid_replicate"] else "invalid"
            print(
                f"[{done}/{total}] {gen_name} rep={rep}: {outcome} n_sub={row['n_participants']} | "
                f"rep_skips={skip_participant_count - rep_skip_start} "
                f"rep_fit_fails={fit_fail_count - rep_fit_fail_start}"
            )

    rep_df = pd.DataFrame(rep_rows)
    rep_path = out_dir / "replicate_bmc_results.csv"
    rep_df.to_csv(rep_path, index=False)
    pct, paths = write_confusion_outputs(rep_df, model_names, out_dir)

    summary_path = out_dir / "run_summary.txt"
    lines = [
        "Model identifiability confusion run",
        f"exp_name: {exp_name}",
        f"fit_npz: {_rel(fit_npz_path)}",
        f"data_inf: {_rel(data_inf_path)}",
        f"n_reps: {args.n_reps}",
        f"seed: {args.seed}",
        f"bads_nit: {args.bads_nit}",
        f"max_nsubs: {args.max_nsubs}",
        f"max_ntrials: {args.max_ntrials}",
        f"max_nsess: {args.max_nsess}",
        f"models_used: {model_names}",
        "models_missing_in_npz: []",
        f"neutral_value: {NEUTRAL}",
        *summary_tail(rep_df, pct, model_names, fit_fail_count=fit_fail_count,
                      skip_participant_count=skip_participant_count,
                      row_sums_heading="Row sums of confusion percent (should be ~100 for rows with valid replicates):"),
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    for path in (rep_path, *paths, summary_path):
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
