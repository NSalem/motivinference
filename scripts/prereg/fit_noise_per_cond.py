"""Preregistered per-condition sigma model fit entrypoint."""

from __future__ import annotations

import argparse
import random as _random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from motinf.prereg.model import (
    build_model_categ,
    fit_model_BADS,
)

GLOBAL_SEED = 42
_random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)


def _trials_ok(n_trials: int, n_params: int) -> bool:
    return n_trials >= n_params + 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preregistered per-condition sigma model fitting.")
    parser.add_argument("--exp-name", default="exp1", help="Experiment subfolder under data/processed/.")
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
    max_nsubs = args.max_nsubs
    bads_nit = args.bads_nit
    data_inf_path = ROOT / "data" / "processed" / args.exp_name / "data_clean_inf.csv"
    out_path = ROOT / "results" / "prereg" / args.exp_name / "model_fit" / "fit_noise_per_cond.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_inf = pd.read_csv(str(data_inf_path))
    df_inf["seqlen"] = df_inf["seqlen"].astype(int)
    participants = np.unique(df_inf["participant"])
    incentives = np.sort(np.unique(df_inf["incentive"].to_numpy()))
    if max_nsubs > 0:
        participants = participants[:max_nsubs]

    categ_specs = [
        {"name": "ang", "parnames": ["sd_ang"]},
        {"name": "inf", "parnames": ["sd_inf"]},
        {"name": "sel", "parnames": ["sd_sel"]},
        # {"name": "ang_inf_sel", "parnames": ["sd_ang", "sd_inf", "sd_sel"]},
    ]
    categ_models = [build_model_categ(s["parnames"]) for s in categ_specs]

    n_p, n_i = len(participants), len(incentives)
    ll_arrays = {s["name"]: np.full((n_p, n_i), np.nan) for s in categ_specs}
    aicc_arrays = {s["name"]: np.full((n_p, n_i), np.nan) for s in categ_specs}
    bic_arrays = {s["name"]: np.full((n_p, n_i), np.nan) for s in categ_specs}
    pars_arrays = {s["name"]: np.empty((n_p, n_i), dtype=object) for s in categ_specs}
    meta_for_save = [{"name": s["name"], "parnames": list(s["parnames"]), "fixed_keys": []} for s in categ_specs]

    for ip, isub in enumerate(participants):
        for ji, inc in enumerate(incentives):
            df_c = df_inf[(df_inf["participant"] == isub) & (df_inf["incentive"] == inc)].reset_index(drop=True)
            for spec, model in zip(categ_specs, categ_models):
                name = spec["name"]
                npc = len(model["parnames"])
                if not _trials_ok(len(df_c), npc):
                    pars_arrays[name][ip, ji] = None
                    continue
                try:
                    print(f"____FITTING sub {ip + 1}/{n_p}, {isub}, inc {inc}, model {name}____")
                    res_c = fit_model_BADS(df=df_c, model=model, fixedpars={}, priors={}, nit=bads_nit, kappa=0.5)
                    
                    pars_arrays[name][ip, ji] = res_c["params"]
                    ll_arrays[name][ip, ji] = res_c["ll"]
                    aicc_arrays[name][ip, ji] = res_c["aicc"]
                    bic_arrays[name][ip, ji] = res_c["bic"]
                except Exception:
                    pars_arrays[name][ip, ji] = None

    save_dict = {
        "participants": participants,
        "incentives": incentives,
        "categ_model_specs": np.array(meta_for_save, dtype=object),
        "categ_models": np.array(categ_models, dtype=object),
        "seed": GLOBAL_SEED,
        "bads_nit": bads_nit,
    }
    for s in categ_specs:
        n = s["name"]
        save_dict[f"pars_{n}"] = pars_arrays[n]
        save_dict[f"ll_{n}"] = ll_arrays[n]
        save_dict[f"aicc_{n}"] = aicc_arrays[n]
        save_dict[f"bic_{n}"] = bic_arrays[n]

    np.savez(str(out_path), **save_dict)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
