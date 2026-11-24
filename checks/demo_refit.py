"""Demo: refit the pre-registered single-noise models to two participants and compare with the bundled fits.

The pre-registered fit (``scripts/prereg/fit_noise_per_cond.py``) fits three single-noise models (sensory, inference
and selection noise: ``ang``, ``inf``, ``sel``) to the Sequence Categorization choices of each participant, separately
in each incentive condition, with BADS from 10 random starting points. This demo repeats exactly that for the first
two participants of Experiment 1 (18 fits): same code, seed, order and settings, read from the bundled fit file. It
prints the refitted and bundled parameter and log-likelihood of every fit and their differences. It writes nothing.

Expected output: 18 rows and "PASS" (log-likelihoods within 0.01 of the bundled ones, parameters within 1%). In the
environment of environment.yml, with 10 or more CPU cores, the refits are identical to the bundled fits (differences
0). With fewer cores they differ in the last digits (in our tests with 8, 4 and 1 cores: log-likelihoods by at most
4e-7, parameters by at most 0.02%). The frozen code runs the 10 restarts of a fit in as many processes as there are
cores, up to 10, and restarts that share a process continue one random stream (data/README.md). Run time: about
30 s (1 min on one core).

Input: ``data/processed/exp1/data_clean_inf.csv``, ``results/prereg/exp1/model_fit/fit_noise_per_cond.npz``.

Usage (from the repository root):
    python checks/demo_refit.py
"""

from __future__ import annotations

import contextlib
import io
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

from motinf.prereg.model import build_model_categ, fit_model_BADS

EXP = "exp1"
N_PARTICIPANTS = 2
MODELS = {"ang": ["sd_ang"], "inf": ["sd_inf"], "sel": ["sd_sel"]}  # order of fit_noise_per_cond.py
TOL_LL = 0.01  # absolute, log-likelihood units
TOL_PAR = 0.01  # relative


def main() -> int:
    bundled = np.load(ROOT / "results/prereg" / EXP / "model_fit/fit_noise_per_cond.npz", allow_pickle=True)
    seed, nit = int(bundled["seed"]), int(bundled["bads_nit"])
    # As at the top of fit_noise_per_cond.py: the seed is set once, then every fit draws its starting points
    # from the same random stream in a fixed order, so the first participants get the same starting points.
    random.seed(seed)
    np.random.seed(seed)

    df = pd.read_csv(ROOT / "data/processed" / EXP / "data_clean_inf.csv")
    df["seqlen"] = df["seqlen"].astype(int)
    participants = np.unique(df["participant"])[:N_PARTICIPANTS]
    incentives = np.sort(np.unique(df["incentive"].to_numpy()))
    assert list(bundled["participants"][:N_PARTICIPANTS]) == list(participants)
    assert list(bundled["incentives"]) == list(incentives)

    rows = []
    t0 = time.time()
    for ip, participant in enumerate(participants):
        for ji, incentive in enumerate(incentives):
            cell = df[(df["participant"] == participant) & (df["incentive"] == incentive)].reset_index(drop=True)
            for name, parnames in MODELS.items():
                print(f"fitting {participant}, incentive {incentive:+.0f}, model {name}", flush=True)
                log = io.StringIO()  # the fitting code prints every restart; keep only failures
                with contextlib.redirect_stdout(log):
                    res = fit_model_BADS(df=cell, model=build_model_categ(parnames), fixedpars={}, priors={},
                                         nit=nit, kappa=0.5)
                for line in log.getvalue().splitlines():
                    if "failed" in line:
                        print("   ", line)
                rows.append({
                    "participant": participant, "incentive": incentive, "model": name, "parameter": parnames[0],
                    "refit": float(res["params"][0]), "bundled": float(bundled[f"pars_{name}"][ip, ji][0]),
                    "ll_refit": float(res["ll"]), "ll_bundled": float(bundled[f"ll_{name}"][ip, ji]),
                })
    elapsed = time.time() - t0

    out = pd.DataFrame(rows)
    out["rel_diff"] = (out["refit"] - out["bundled"]).abs() / out["bundled"].abs()
    out["ll_diff"] = (out["ll_refit"] - out["ll_bundled"]).abs()
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.4g}".format):
        print(out.to_string(index=False))
    ok = bool((out["ll_diff"] <= TOL_LL).all() and (out["rel_diff"] <= TOL_PAR).all())
    print(f"\n{len(out)} fits in {elapsed:.0f} s; largest differences: log-likelihood {out['ll_diff'].max():.2g}, "
          f"parameter {100 * out['rel_diff'].max():.2g}%")
    print("PASS" if ok else f"FAIL: differences above the tolerances (log-likelihood {TOL_LL}, parameter {TOL_PAR:.0%})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
