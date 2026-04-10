"""Preregistered per-participant incentive-channel categorization fits."""

from __future__ import annotations

import argparse
import random as _random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

GLOBAL_SEED = 42

_random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

from motinf.prereg.model import (
    build_model_categ,
    fit_model_BADS,
    priors_categ,
)


def build_categ_specs() -> list[dict]:
    return [
        {"name": "ang_v", "label": "sd_ang + sd_ang_v (single incentive on ang)", "parnames": ["sd_ang", "sd_ang_v", "sd_inf", "sd_sel"], "fixedpars": {}},
        {"name": "inf_v", "label": "sd_inf + sd_inf_v (single incentive on inf)", "parnames": ["sd_ang", "sd_inf", "sd_inf_v", "sd_sel"], "fixedpars": {}},
        {"name": "sel_v", "label": "sd_sel + sd_sel_v (single incentive on sel)", "parnames": ["sd_ang", "sd_inf", "sd_sel", "sd_sel_v"], "fixedpars": {}},
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preregistered incentive-channel model fitting.")
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
    use_priors_categ = True
    priors_c = priors_categ if use_priors_categ else {}

    exp_name = args.exp_name
    data_inf_path = ROOT / "data" / "processed" / exp_name / "data_clean_inf.csv"
    out_path = ROOT / "results" / "prereg" / exp_name / "model_fit" / "fit_inc_effects.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    categ_specs = build_categ_specs()
    categ_models = [build_model_categ(s["parnames"]) for s in categ_specs]
    n_categ = len(categ_specs)

    df_inf = pd.read_csv(str(data_inf_path))
    df_inf["seqlen"] = df_inf["seqlen"].astype(int)

    participants = np.unique(df_inf["participant"])
    if max_nsubs > 0:
        participants = participants[:max_nsubs]

    nsubs = len(participants)
    pars_categ = np.empty((n_categ, nsubs), dtype=object)
    ll_categ = np.zeros((n_categ, nsubs))
    aicc_categ = np.zeros((n_categ, nsubs))
    bic_categ = np.zeros((n_categ, nsubs))

    for nsub, isub in enumerate(participants):
        df_c = df_inf[df_inf["participant"] == isub].reset_index(drop=True)
        t0 = time.time()

        for mi, spec in enumerate(categ_specs):
            print(f"____FITTING {nsub + 1}/{nsubs}, {isub}, model {spec['name']}____")
            model = categ_models[mi]
            fp = dict(spec.get("fixedpars", {}))
            res = fit_model_BADS(df=df_c, model=model, fixedpars=fp, priors=priors_c, nit=bads_nit, kappa=0.5)
            pars_categ[mi, nsub] = res["params"]
            ll_categ[mi, nsub] = res["ll"]
            aicc_categ[mi, nsub] = res["aicc"]
            bic_categ[mi, nsub] = res["bic"]

        print(f"  BADS block {time.time() - t0:.2f}s | categ_priors={use_priors_categ}")

    meta = [
        {"name": s["name"], "label": s["label"], "parnames": list(s["parnames"]), "fixedpars_keys": sorted(list(s.get("fixedpars", {}).keys()))}
        for s in categ_specs
    ]

    np.savez(
        str(out_path),
        participants=participants,
        categ_model_specs=np.array(meta, dtype=object),
        categ_models=np.array(categ_models, dtype=object),
        pars_categ=pars_categ,
        ll_categ=ll_categ,
        aicc_categ=aicc_categ,
        bic_categ=bic_categ,
        categ_model_names=np.array([s["name"] for s in categ_specs], dtype=object),
        seed=GLOBAL_SEED,
        bads_nit=bads_nit,
        use_priors_categ=use_priors_categ,
        script="fit_incentive_effects_prereg",
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
