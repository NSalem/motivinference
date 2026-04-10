"""Preregistered model."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import os

import numpy as np
from pybads import BADS
from scipy.special import iv
from scipy.stats import beta, gamma, norm
from threadpoolctl import threadpool_limits

priors_categ = {
    "sd_ang": gamma(a=1.1, scale=np.deg2rad(2)),
    "sd_ang_v": gamma(a=1.1, scale=np.deg2rad(2)),
    "sd_inf": gamma(a=2, scale=0.5),
    "sd_inf_v": gamma(a=2, scale=0.5),
    "sd_sel": gamma(a=1.1, scale=2),
    "sd_sel_v": gamma(a=1.1, scale=2),
    "plapse_choice": beta(a=1, b=15),
    "choice_bias": norm(loc=0, scale=0.5),
}

param_config_categ = {
    "sd_ang": {"lb": np.deg2rad(0.001), "ub": np.deg2rad(50), "plb": np.deg2rad(0.1), "pub": np.deg2rad(15), "x0": np.deg2rad(4)},
    "sd_ang_v": {"lb": np.deg2rad(0.001), "ub": np.deg2rad(50), "plb": np.deg2rad(0.1), "pub": np.deg2rad(15), "x0": np.deg2rad(4)},
    "sd_inf": {"lb": 1e-6, "ub": 5, "plb": 0.1, "pub": 1.5, "x0": 0.5},
    "sd_inf_v": {"lb": 1e-6, "ub": 5, "plb": 0.1, "pub": 1.5, "x0": 0.5},
    "sd_sel": {"lb": 1e-6, "ub": 10, "plb": 0.1, "pub": 4, "x0": 1},
    "sd_sel_v": {"lb": 1e-6, "ub": 10, "plb": 0.1, "pub": 4, "x0": 1},
    "plapse_choice": {"lb": 1e-10, "ub": 0.1, "plb": 0.005, "pub": 0.05, "x0": 0.01},
    "choice_bias": {"lb": -5, "ub": 5, "plb": -1, "pub": 1, "x0": 0},
}


def df_to_pack(df):
    seqlen = df["seqlen"].to_numpy(int, copy=False)
    choice = df["choice"].to_numpy(copy=False) if "choice" in df else None
    incentive = df["incentive"].to_numpy(copy=False) if "incentive" in df else None
    seqang_cols = [f"seqang_noisy_{i}" for i in range(1, 13)]
    seqang = np.column_stack([df[c].to_numpy(copy=False) for c in seqang_cols])
    n_rows, n_cols = seqang.shape
    cols = np.arange(n_cols)[None, :]
    valid = cols < seqlen[:, None]
    seqang[~valid] = np.nan
    return {"seqang": seqang, "seqlen": seqlen, "choice": choice, "incentive": incentive, "N": n_rows, "T": n_cols}


def psychofun(pars, data, kappa=0.5, sigstr=1, bounds_config=param_config_categ):
    seqang = data["seqang"]
    seqlen = data["seqlen"]
    choice = data["choice"]
    incentive = data["incentive"] if data["incentive"] is not None else np.zeros(len(choice))

    sd_ang = pars.get("sd_ang", 0.0)
    sd_inf = pars.get("sd_inf", 0.0)
    sd_sel = pars.get("sd_sel", 0.0)
    sd_ang_v = pars.get("sd_ang_v", sd_ang)
    sd_inf_v = pars.get("sd_inf_v", sd_inf)
    sd_sel_v = pars.get("sd_sel_v", sd_sel)
    plapse = pars.get("plapse_choice", 0.0)
    choice_bias = pars.get("choice_bias", 0.0)

    sd_ang_t = np.where(incentive == 1, sd_ang_v, np.where(incentive == -1, sd_ang_v, sd_ang))
    sd_inf_t = np.where(incentive == 1, sd_inf_v, np.where(incentive == -1, sd_inf_v, sd_inf))
    sd_sel_t = np.where(incentive == 1, sd_sel_v, np.where(incentive == -1, sd_sel_v, sd_sel))

    sd_ang_vec = sd_ang_t[:, None]
    mu_llr = 2 * kappa * sigstr * np.exp(-2 * sd_ang_vec**2) * np.cos(2 * (seqang - np.pi * 3 / 4))
    s2_llr = 2 * (kappa**2) * (sigstr**2) * (1 + np.exp(-8 * sd_ang_vec**2) * np.cos(4 * (seqang - np.pi * 3 / 4)))

    diff = np.clip(s2_llr - mu_llr**2, 0, None)
    sd_llr = np.sqrt(diff)
    mu_llr[np.isnan(mu_llr)] = 0.0
    sd_llr[np.isnan(sd_llr)] = 0.0

    xpost = np.nansum(mu_llr, axis=1)
    spost = np.sqrt(np.nansum(sd_llr**2, axis=1) + (seqlen - 1) * (sd_inf_t**2) + (sd_sel_t**2))
    spost = np.clip(spost, 1e-20, np.inf)

    ppost = norm.cdf(xpost + choice_bias, 0, spost)
    ppost = (1 - plapse) * ppost + plapse / 2.0
    p = ppost * (choice == 1) + (1 - ppost) * (choice == -1)
    p = np.clip(p, 1e-20, 1)

    return p,spost,xpost


def get_ll(theta, parnames, data, fixedpars=None, kappa=0.5, sigstr=1, bounds_config=param_config_categ):
    fixedpars = {} if fixedpars is None else fixedpars
    pars = {"sd_ang": 0.0, "sd_inf": 0.0, "sd_sel": 0.0, "plapse_choice": 0.0, "choice_bias": 0.0}
    pars.update(dict(zip(parnames, theta)))
    pars.update(fixedpars)
    return np.sum(np.log(psychofun(pars, data, kappa, sigstr, bounds_config=bounds_config)[0]))


def get_log_posterior(theta, parnames, data, fixedpars=None, priors=None, kappa=0.5, bounds_config=param_config_categ):
    fixedpars = {} if fixedpars is None else fixedpars
    priors = {} if priors is None else priors
    ll = get_ll(theta, parnames, data, fixedpars, kappa=kappa, bounds_config=bounds_config)
    log_prior = 0.0
    for i, pname in enumerate(parnames):
        if pname in priors:
            lp = priors[pname].logpdf(theta[i])
            if np.isnan(lp) or np.isinf(lp):
                return -np.inf
            log_prior += lp
    return ll + log_prior


def _run_bads_with_limits(theta0_i, lb, ub, plb, pub, options, parnames, data, fixedpars, priors, kappa, worker_id):
    with threadpool_limits(limits=1):
        bads = BADS(
            lambda theta: -get_log_posterior(theta, parnames, data, fixedpars, priors, kappa=kappa),
            theta0_i,
            lb,
            ub,
            plb,
            pub,
            options=options,
        )
        res_i = bads.optimize()
        return {"worker_id": worker_id, "x": np.asarray(res_i.x), "fval": float(res_i.fval)}


def fit_model_BADS(
    df,
    parnames=None,
    lb=None,
    ub=None,
    plb=None,
    pub=None,
    x0=None,
    fixedpars=None,
    priors=None,
    nit=1,
    kappa=0.5,
    options=None,
    model=None,
):
    if model is not None:
        parnames = model["parnames"]
        lb = model["lb"]
        ub = model["ub"]
        plb = model.get("plb", plb)
        pub = model.get("pub", pub)
        x0 = model.get("x0", x0)

    if parnames is None or lb is None or ub is None:
        raise ValueError("Provide either `model` or explicit `parnames`, `lb`, and `ub`.")

    fixedpars = {} if fixedpars is None else fixedpars
    priors = {} if priors is None else priors
    options = {"uncertainty_handling": False, "display": "off"} if options is None else options

    if plb is None:
        plb = np.array(lb) + (np.array(ub) - np.array(lb)) * 0
    if pub is None:
        pub = np.array(lb) + (np.array(ub) - np.array(lb)) * 0.8

    ntrials = len(df)
    data = df_to_pack(df)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    plb = np.asarray(plb, dtype=float)
    pub = np.asarray(pub, dtype=float)
    nparams = len(lb)
    theta0 = [np.random.uniform(low=plb, high=pub) for _ in range(nit)] if x0 is None else [x0 for _ in range(nit)]

    results = []
    parallel_failed = False
    if nit > 1:
        try:
            with ProcessPoolExecutor(max_workers=min(nit, os.cpu_count() or 1)) as ex:
                futs = [
                    ex.submit(_run_bads_with_limits, theta0[i], lb, ub, plb, pub, options, parnames, data, fixedpars, priors, kappa, i)
                    for i in range(nit)
                ]
                for fut in as_completed(futs):
                    res_i = fut.result()
                    results.append(res_i)
                    print(f"Results for iteration {res_i['worker_id']}: {np.round(res_i['x'], 4)}")
        except (PermissionError, OSError):
            parallel_failed = True
    else:
        parallel_failed = True

    if parallel_failed:
        results = []
        for i in range(nit):
            res_i = _run_bads_with_limits(theta0[i], lb, ub, plb, pub, options, parnames, data, fixedpars, priors, kappa, i)
            results.append(res_i)
            print(f"Results for iteration {res_i['worker_id']}: {np.round(res_i['x'], 4)}")

    res = min(results, key=lambda r: r["fval"])
    params_final = np.asarray(res["x"], dtype=float)
    ll = get_ll(params_final, parnames, data, fixedpars, kappa=kappa)
    print(f"Returned parameter vector: {np.round(params_final, 4)}")
    print(f"Log-likelihood at solution: {ll:.4f}")
    aic = 2 * nparams - 2 * ll
    aicc = aic + (2 * nparams * (nparams + 1)) / (ntrials - nparams - 1)
    bic = -2 * ll + nparams * np.log(ntrials)
    return {"params": params_final, "ll": ll, "aicc": aicc, "bic": bic}


def build_model_categ(parnames: list[str]) -> dict:
    return {
        "parnames": parnames,
        "lb": [param_config_categ[p]["lb"] for p in parnames],
        "ub": [param_config_categ[p]["ub"] for p in parnames],
        "plb": [param_config_categ[p]["plb"] for p in parnames],
        "pub": [param_config_categ[p]["pub"] for p in parnames],
        "x0": [param_config_categ[p]["x0"] for p in parnames],
    }
