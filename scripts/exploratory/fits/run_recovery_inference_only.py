# -*- coding: utf-8 -*-
"""Parameter recovery for the inference-only model (Supplementary Figure 2b).

For every participant and incentive condition, choices are simulated on the participant's trials with the inference
noise ``sd_inf`` fitted by the pre-registered fit (``pars_inf``; the other noise sources at zero), and ``sd_inf`` is
refitted to them without priors (BADS from ``--bads-nit`` random starting points; until 16 Nov 2026 all restarts
started from the same point).

Inputs: ``data/processed/<exp>/data_clean_inf.csv``, ``results/prereg/<exp>/model_fit/fit_noise_per_cond.npz``.
Outputs (``results/exploratory/<exp>/modeling/recovery_noise_per_cond/``): ``recovery_runs.csv`` (generative and
recovered ``sd_inf`` per participant and condition; read by ``sfig2.py``) and ``recovery_summary.json``.

Usage (from the repository root), with the paper's settings (one simulation per cell, seed 42, 10 BADS starts):
    python scripts/exploratory/fits/run_recovery_inference_only.py --exp-name exp1
"""

from __future__ import annotations

import argparse
import json
import random as _random
from pathlib import Path

import numpy as np
import pandas as pd

from motinf.exploratory.cohorts import apply_row_limits
from motinf.exploratory.model import create_categ_model, run_fit_categ, simulate_choice, trials_ok

ROOT = Path(__file__).resolve().parents[3]
GLOBAL_SEED = 42


def _rel(p) -> str:
    """A path as recorded in the run summary: relative to the repository when inside it."""
    p = Path(p).resolve()
    return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Parameter recovery: simulate from prereg NPZ `pars_inf`, refit with dev BADS (no priors)."
        )
    )
    p.add_argument("--exp-name", default="exp1", help="Experiment under data/processed/ and results/prereg/.")
    p.add_argument("--n-reps", type=int, default=1, help="Stochastic simulation replicates per cell.")
    p.add_argument("--seed", type=int, default=GLOBAL_SEED)
    p.add_argument("--bads-nit", type=int, default=10, help="BADS restarts.")
    p.add_argument("--max-nsubs", type=int, default=0, help="0 = all participants listed in the NPZ.")
    p.add_argument(
        "--max-ntrials",
        type=int,
        default=0,
        help="0 = use every trial in the cell; otherwise keep rows with trialN <= max-ntrials.",
    )
    p.add_argument(
        "--max-nsess",
        type=int,
        default=0,
        help="0 = use every session; otherwise keep rows with session <= max-nsess.",
    )
    p.add_argument(
        "--results-npz",
        default="",
        help="Override path to fit_noise_per_cond.npz (default: results/prereg/<exp>/model_fit/fit_noise_per_cond.npz).",
    )
    p.add_argument(
        "--out-dir",
        default="",
        help="Output directory (default: results/exploratory/<exp>/modeling/recovery_noise_per_cond).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _random.seed(args.seed)
    np.random.seed(args.seed)

    data_path = ROOT / "data" / "processed" / args.exp_name / "data_clean_inf.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data: {data_path}")

    npz_path = (
        Path(args.results_npz).expanduser().resolve()
        if args.results_npz.strip()
        else ROOT / "results" / "prereg" / args.exp_name / "model_fit" / "fit_noise_per_cond.npz"
    )
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing prereg fits: {npz_path}")

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir.strip()
        else ROOT / "results" / "exploratory" / args.exp_name / "modeling" / "recovery_noise_per_cond"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    fit = np.load(npz_path, allow_pickle=True)
    participants_fit = np.asarray(fit["participants"])
    incentives_fit = np.asarray(fit["incentives"])
    pars_inf = fit["pars_inf"]

    df_inf = pd.read_csv(str(data_path))
    df_inf["seqlen"] = df_inf["seqlen"].astype(int)
    df_inf = apply_row_limits(df_inf, args.max_ntrials, args.max_nsess)

    model = create_categ_model(["sd_inf"])
    parnames = model["parnames"]
    npar = len(parnames)
    fixedpars: dict = {}
    priors: dict = {}

    rows: list[dict] = []
    rep_global = 0

    n_p = len(participants_fit)
    for ip, sub in enumerate(participants_fit):
        if args.max_nsubs > 0 and ip >= args.max_nsubs:
            break
        for ji, inc in enumerate(incentives_fit):
            raw = pars_inf[ip, ji]
            if raw is None:
                continue
            sd_inf_true = float(np.asarray(raw, dtype=float).ravel()[0])
            if not np.isfinite(sd_inf_true):
                continue

            df_cell = df_inf[
                (df_inf["participant"] == sub) & (df_inf["incentive"] == inc)
            ].reset_index(drop=True)
            n_trial = len(df_cell)
            if n_trial == 0:
                continue
            if not trials_ok(n_trial, npar):
                continue

            base_cols = [c for c in df_cell.columns if c != "choice"]
            gen_params = {"sd_inf": sd_inf_true}

            for rep in range(args.n_reps):
                rng = np.random.default_rng(args.seed + rep_global * 1_000_003 + int(ip) * 97 + int(ji))
                rep_global += 1

                np.random.seed(int(rng.integers(0, 2**31 - 1)))
                choice_sim, _, _, _ = simulate_choice(gen_params, df_cell)

                df_sim = df_cell[base_cols].copy()
                df_sim["choice"] = choice_sim

                try:
                    print(f"FITTING sub {ip + 1}/{n_p}, {sub}, inc {inc}, rep {rep}/{args.n_reps}____")

                    res = run_fit_categ(df_sim, model, fixedpars, priors, args.bads_nit)
                    #print generative, then fitted
                    print(f"Generative: {gen_params['sd_inf']}")
                    print(f"Fitted: {res['params'][0]}")
                    theta = np.asarray(res["params"], dtype=float).ravel()
                    sd_fit = float(theta[0])
                    ll = float(res["ll"])
                    aicc = float(res["aicc"])
                    ok = np.isfinite(sd_fit) and np.isfinite(ll)
                except Exception as exc:
                    sd_fit = np.nan
                    ll = np.nan
                    aicc = np.nan
                    ok = False
                    err_msg = str(exc)
                else:
                    err_msg = ""

                rows.append(
                    {
                        "participant": sub,
                        "incentive": inc,
                        "rep": rep,
                        "sd_inf_true": sd_inf_true,
                        "sd_inf_fit": sd_fit,
                        "bias": sd_fit - sd_inf_true if ok else np.nan,
                        "ll": ll,
                        "aicc": aicc,
                        "n_trials": n_trial,
                        "ok": ok,
                        "error": err_msg,
                    }
                )

    if not rows:
        raise RuntimeError(
            "No recovery rows produced (check NPZ pars_inf, trial counts, and n_params)."
        )

    out_csv = out_dir / "recovery_runs.csv"
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_csv, index=False)

    sub_ok = df_out[df_out["ok"]]
    summary = {
        "npz_path": _rel(npz_path),
        "max_ntrials": int(args.max_ntrials),
        "max_nsess": int(args.max_nsess),
        "subsample_trials": 0,  # every trial of each cell is used
        "n_rows": len(df_out),
        "n_ok": int(sub_ok["ok"].sum()),
        "mae_sd_inf": float(np.nanmean(np.abs(sub_ok["sd_inf_fit"] - sub_ok["sd_inf_true"])))
        if len(sub_ok)
        else float("nan"),
        "rmse_sd_inf": float(
            np.sqrt(np.nanmean((sub_ok["sd_inf_fit"] - sub_ok["sd_inf_true"]) ** 2))
        )
        if len(sub_ok)
        else float("nan"),
    }
    if len(sub_ok) >= 2:
        summary["corr_true_fit"] = float(
            np.corrcoef(sub_ok["sd_inf_true"], sub_ok["sd_inf_fit"])[0, 1]
        )
    else:
        summary["corr_true_fit"] = float("nan")

    summary_path = out_dir / "recovery_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved: {out_csv}")
    print(f"Saved: {summary_path}")
    print(summary)


if __name__ == "__main__":
    main()
