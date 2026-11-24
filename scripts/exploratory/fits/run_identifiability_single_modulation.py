# -*- coding: utf-8 -*-
"""Model recovery for the single-modulation models (Supplementary Figure 4b).

In each simulation, choices are simulated for every participant's trial sequences from one of the three models (an
incentive effect on sensory, inference or selection noise: ``sd_ang``, ``sd_inf``, ``sd_sel`` drawn from the priors
of ``priors_categ``, the active ``motiv_*`` from N(``--motiv-mean``, ``--motiv-sd``) = N(0.1, 0.3); under incentive
each SD is ``sd_X * exp(-motiv_X)``), all three models are refitted (BADS, ``--bads-nit`` starts, no priors), and
group Bayesian model selection picks the recovered model.

Input: ``data/processed/<exp>/data_clean_inf.csv``.
Outputs (``results/exploratory/<exp>/modeling/model_motiv_identifiability/``): ``confusion_matrix_percent.csv``
(read by ``sfig4.py``), ``confusion_matrix_counts.csv``, ``replicate_bmc_results.csv``, the protected exceedance
probabilities, parameter-recovery tables and ``run_summary.txt`` (settings).

Usage (from the repository root), with the paper's settings (seed 42, 5 BADS starts):
    python scripts/exploratory/fits/run_identifiability_single_modulation.py --exp-name exp1 --n-reps 20
"""

from __future__ import annotations

import argparse
import json
import random as _random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from motinf.exploratory.cohorts import apply_row_limits
from motinf.exploratory.identifiability import replicate_bmc_row, summary_tail, write_confusion_outputs
from motinf.exploratory.model import create_categ_model, priors_categ, run_fit_categ, simulate_choice, subset_priors
from motinf.exploratory.model import trials_ok

ROOT = Path(__file__).resolve().parents[3]
# The three single-modulation models: the three noise SDs and one incentive effect.
MODEL_SPECS = {
    "ang_motiv": ["sd_ang", "sd_inf", "sd_sel", "motiv_ang"],
    "inf_motiv": ["sd_ang", "sd_inf", "sd_sel", "motiv_inf"],
    "sel_motiv": ["sd_ang", "sd_inf", "sd_sel", "motiv_sel"],
}


def _rel(p) -> str:
    """A path as recorded in the run summary: relative to the repository when inside it."""
    p = Path(p).resolve()
    return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)


def sample_positive_sd(prior_rv, rng: np.random.Generator, max_tries: int = 100) -> float:
    """Single draw from prior, resampling until strictly positive and finite."""
    for _ in range(max_tries):
        x = float(prior_rv.rvs(random_state=rng))
        if np.isfinite(x) and x > 0:
            return x
    raise RuntimeError("Could not sample a positive SD from prior after many tries.")


def draw_generative_params(parnames: list[str], motiv_mean: float, motiv_sd: float,
                           rng: np.random.Generator) -> dict[str, float]:
    """Generative parameters of a model: ``sd_ang``, ``sd_inf`` and ``sd_sel`` from ``priors_categ`` (in that order),
    then its incentive effect (the last of ``parnames``) from N(``motiv_mean``, ``motiv_sd``)."""
    params = {k: sample_positive_sd(priors_categ[k], rng) for k in ("sd_ang", "sd_inf", "sd_sel")}
    params[parnames[-1]] = float(norm(loc=motiv_mean, scale=motiv_sd).rvs(random_state=rng))
    return {p: float(params[p]) for p in parnames}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Identifiability confusion for pooled categorization with motiv-space models "
            "(simulate, refit all candidates, GroupBMC per replicate)."
        )
    )
    p.add_argument("--exp-name", default="exp1")
    p.add_argument("--n-reps", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bads-nit", type=int, default=5)
    p.add_argument("--max-nsubs", type=int, default=0, help="0 = all participants.")
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
        "--out-dir",
        default="",
        help=(
            "Override output directory. "
            "Default: results/exploratory/<exp>/modeling/model_motiv_identifiability"
        ),
    )
    p.add_argument(
        "--use-priors",
        action="store_true",
        help="Use priors_categ subset as fitting priors (default: no priors in fitting step).",
    )
    p.add_argument(
        "--motiv-mean",
        type=float,
        default=0.1,
        help="Mean of Gaussian motiv (scipy norm loc) for generative draws.",
    )
    p.add_argument(
        "--motiv-sd",
        type=float,
        default=0.3,
        help="Std dev of Gaussian motiv (scipy norm scale, not variance) for generative draws.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _random.seed(args.seed)
    np.random.seed(args.seed)

    if float(args.motiv_sd) <= 0:
        raise ValueError("--motiv-sd must be positive.")

    exp_name = args.exp_name
    data_inf_path = ROOT / "data" / "processed" / exp_name / "data_clean_inf.csv"
    if args.out_dir.strip():
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = ROOT / "results" / "exploratory" / exp_name / "modeling" / "model_motiv_identifiability"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_inf_path.exists():
        raise FileNotFoundError(f"Missing data: {data_inf_path}")

    df_inf = pd.read_csv(data_inf_path)
    df_inf["seqlen"] = df_inf["seqlen"].astype(int)
    df_inf = apply_row_limits(df_inf, args.max_ntrials, args.max_nsess)
    df_inf["participant_key"] = df_inf["participant"].astype(str)

    model_names = list(MODEL_SPECS)
    models = {name: create_categ_model(parnames) for name, parnames in MODEL_SPECS.items()}
    fit_priors = {name: subset_priors(priors_categ, m["parnames"]) if args.use_priors else {}
                  for name, m in models.items()}

    participants = np.asarray(pd.unique(df_inf["participant_key"]), dtype=object)
    if args.max_nsubs > 0:
        participants = participants[: args.max_nsubs]
    if participants.size == 0:
        raise ValueError("No participants available after filtering.")

    rep_rows: list[dict] = []
    param_rows: list[dict] = []
    fit_fail_count = 0
    skip_participant_count = 0

    total = len(model_names) * args.n_reps
    done = 0

    for igen, gen_name in enumerate(model_names):
        print(f"\n=== Generative model: {gen_name} ({args.n_reps} replicates) ===")

        for rep in range(args.n_reps):
            done += 1
            aicc = np.full((len(participants), len(model_names)), np.nan, dtype=float)

            for ip, sub in enumerate(participants):
                # Seeds per generative model, replicate and participant.
                rng_draw = np.random.default_rng(int(args.seed) + rep * 7919 + ip * 104729 + igen * 17)
                params_gen = draw_generative_params(MODEL_SPECS[gen_name], float(args.motiv_mean),
                                                    float(args.motiv_sd), rng_draw)
                df_sub = df_inf[df_inf["participant_key"] == sub].reset_index(drop=True)

                np.random.seed(args.seed + rep * 100003 + ip * 1009 + igen * 17)
                try:
                    sim_choice, _, _, _ = simulate_choice(params_gen, df_sub)
                except Exception:
                    fit_fail_count += 1
                    continue

                df_sim = df_sub.copy()
                df_sim["choice"] = sim_choice

                for jm, cand_name in enumerate(model_names):
                    print(
                        f"Generative model [{igen + 1}/{len(model_names)}], "
                        f"replicate [{rep + 1}/{args.n_reps}], "
                        f"participant [{ip + 1}/{len(participants)}], "
                        f"candidate model [{jm + 1}/{len(model_names)}]"
                    )
                    m = models[cand_name]
                    if not trials_ok(len(df_sim), len(m["parnames"])):
                        continue
                    try:
                        res = run_fit_categ(df_sim, m, {}, fit_priors[cand_name], args.bads_nit)
                        if not np.isfinite(res["aicc"]):
                            fit_fail_count += 1
                            continue
                        aicc[ip, jm] = float(res["aicc"])
                        theta_rec = np.asarray(res["params"], dtype=float).ravel()
                        param_rows.append(
                            {
                                "replicate": rep,
                                "generative_model": gen_name,
                                "participant": sub,
                                "candidate_model": cand_name,
                                "generative_params_json": json.dumps(params_gen),
                                "recovered_params_json": json.dumps(
                                    {name: float(theta_rec[k]) for k, name in enumerate(m["parnames"])}
                                ),
                                "ll": float(res["ll"]),
                                "aicc": float(res["aicc"]),
                                "bic": float(res["bic"]),
                            }
                        )
                    except Exception:
                        fit_fail_count += 1

            row = replicate_bmc_row(gen_name, rep, aicc, model_names)
            rep_rows.append(row)
            outcome = f"recovered={row['recovered_model']}" if row["valid_replicate"] else "invalid"
            print(f"[{done}/{total}] {gen_name} rep={rep}: {outcome} n_sub={row['n_participants']}")

    rep_df = pd.DataFrame(rep_rows)
    rep_path = out_dir / "replicate_bmc_results.csv"
    rep_df.to_csv(rep_path, index=False)
    pct, paths = write_confusion_outputs(rep_df, model_names, out_dir)

    if param_rows:
        pr_df = pd.DataFrame(param_rows)
        paths.append(out_dir / "parameter_recovery_fits.csv")
        pr_df.to_csv(paths[-1], index=False)

        long_rows = []
        for _, row in pr_df.iterrows():
            g = json.loads(row["generative_params_json"])
            r = json.loads(row["recovered_params_json"])
            for pname in [k for k in g if k in r]:  # dict order: the same rows in the same order on every run
                long_rows.append(
                    {
                        "replicate": row["replicate"],
                        "generative_model": row["generative_model"],
                        "candidate_model": row["candidate_model"],
                        "participant": row["participant"],
                        "parameter": pname,
                        "value_true": float(g[pname]),
                        "value_recovered": float(r[pname]),
                    }
                )
        paths.append(out_dir / "parameter_recovery_long.csv")
        pd.DataFrame(long_rows).to_csv(paths[-1], index=False)

    summary_path = out_dir / "run_summary.txt"
    lines = [
        "Model identifiability (pooled categorization, motiv-space, no estimation task)",
        f"exp_name: {exp_name}",
        f"data_inf: {_rel(data_inf_path)}",
        f"n_reps: {args.n_reps}",
        f"seed: {args.seed}",
        f"bads_nit: {args.bads_nit}",
        f"max_nsubs: {args.max_nsubs}",
        f"max_ntrials: {args.max_ntrials}",
        f"max_nsess: {args.max_nsess}",
        f"use_priors: {args.use_priors}",
        "generation: sd_* from priors_categ; active motiv_* ~ N(motiv_mean, motiv_sd)",
        f"motiv_mean: {args.motiv_mean}",
        f"motiv_sd: {args.motiv_sd}",
        "likelihood_mapping: sd_*_v = sd_* * exp(-motiv_*) for the channel's motiv (implicit sd_*_v)",
        f"models: {model_names}",
        *summary_tail(rep_df, pct, model_names, fit_fail_count=fit_fail_count,
                      skip_participant_count=skip_participant_count, row_sums_heading="Row sums of confusion percent:"),
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    for path in (rep_path, *paths, summary_path):
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
