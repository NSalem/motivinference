"""Fit the single-noise inference model separately to fast and slow trials (Figure 5C, Supplementary Table 9).

Trials are split by each participant's median RT into fast (rt <= median) and slow (rt > median). Within each
participant and RT bin, the 3-parameter model (sigma_inf for neutral, gain and loss) is fitted with BADS
(``--bads-nit`` restarts, no priors).

Output: ``results/exploratory/<exp>/modeling/fit_files/fit_inf_rt_split.npz`` (read by ``rt_analysis.py``). The
starting points of all fits come from one random stream seeded once per run, so the bundled fits are reproduced by
one run per experiment.

Usage (from repo root), once per experiment:
    python scripts/exploratory/fits/fit_inf_per_cond_rt_split.py --exp-name exp1
"""

from __future__ import annotations

import argparse
import random as _random
from pathlib import Path

import numpy as np
import pandas as pd

from motinf.exploratory.model import create_categ_model, run_fit_categ, trials_ok

ROOT = Path(__file__).resolve().parents[3]
GLOBAL_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-noise inference model fitted to fast and slow trials separately.")
    parser.add_argument("--exp-name", default="exp1", help="Experiment subfolder under data/processed/.")
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory (default: results/exploratory/<exp>/modeling/fit_files).",
    )
    parser.add_argument(
        "--max-nsubs",
        type=int,
        default=0,
        help="Maximum number of participants to fit (0 = all, in stable sorted order).",
    )
    parser.add_argument(
        "--bads-nit",
        type=int,
        default=10,
        help="Number of BADS random-restart optimizations per fit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _random.seed(args.seed)
    np.random.seed(args.seed)
    max_nsubs = args.max_nsubs
    bads_nit = args.bads_nit

    data_inf_path = ROOT / "data" / "processed" / args.exp_name / "data_clean_inf.csv"
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir.strip()
        else ROOT / "results" / "exploratory" / args.exp_name / "modeling" / "fit_files"
    )
    out_path = out_dir / "fit_inf_rt_split.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_inf = pd.read_csv(str(data_inf_path))
    df_inf["seqlen"] = df_inf["seqlen"].astype(int)
    df_inf["rt"] = pd.to_numeric(df_inf["rt"], errors="coerce")
    df_inf = df_inf[df_inf["rt"].notna() & (df_inf["rt"] > 0)].copy()

    rt_median = df_inf.groupby("participant")["rt"].transform("median")
    df_inf["rt_bin"] = np.where(df_inf["rt"] <= rt_median, "fast", "slow")

    participants = np.unique(df_inf["participant"])
    if max_nsubs > 0:
        participants = participants[:max_nsubs]

    rt_bins = np.array(["fast", "slow"], dtype=object)
    model_spec = {"name": "inf_only_rt_split", "parnames": ["sd_inf", "sd_inf_g", "sd_inf_l"]}
    model = create_categ_model(model_spec["parnames"])
    n_params = len(model["parnames"])

    n_p, n_b = len(participants), len(rt_bins)
    pars_inf_rt_split = np.empty((n_p, n_b), dtype=object)
    ll_inf_rt_split = np.full((n_p, n_b), np.nan)
    aicc_inf_rt_split = np.full((n_p, n_b), np.nan)
    bic_inf_rt_split = np.full((n_p, n_b), np.nan)
    n_trials_rt_split = np.zeros((n_p, n_b), dtype=int)

    for ip, isub in enumerate(participants):
        for jb, rt_bin in enumerate(rt_bins):
            df_c = df_inf[(df_inf["participant"] == isub) & (df_inf["rt_bin"] == rt_bin)].reset_index(drop=True)
            n_trials_rt_split[ip, jb] = len(df_c)
            if not trials_ok(len(df_c), n_params):
                pars_inf_rt_split[ip, jb] = None
                continue

            try:
                print(f"____FITTING sub {ip + 1}/{n_p}, {isub}, rt_bin {rt_bin}, model inf_only____")
                res_c = run_fit_categ(df_c, model, bads_nit=bads_nit)
                pars_inf_rt_split[ip, jb] = res_c["params"]
                ll_inf_rt_split[ip, jb] = res_c["ll"]
                aicc_inf_rt_split[ip, jb] = res_c["aicc"]
                bic_inf_rt_split[ip, jb] = res_c["bic"]
            except Exception:
                pars_inf_rt_split[ip, jb] = None

    save_dict = {
        "participants": participants,
        "rt_bins": rt_bins,
        "model_name": model_spec["name"],
        "model_parnames": np.array(model_spec["parnames"], dtype=object),
        "model_spec": np.array(
            [{"name": model_spec["name"], "parnames": list(model_spec["parnames"]), "fixed_keys": []}],
            dtype=object,
        ),
        "model": np.array(model, dtype=object),
        "pars_inf_rt_split": pars_inf_rt_split,
        "ll_inf_rt_split": ll_inf_rt_split,
        "aicc_inf_rt_split": aicc_inf_rt_split,
        "bic_inf_rt_split": bic_inf_rt_split,
        "n_trials_rt_split": n_trials_rt_split,
        "seed": args.seed,
        "bads_nit": bads_nit,
        "script": "fit_inf_only_rt_split",
    }

    np.savez(str(out_path), **save_dict)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
