"""Full-model fit of one participant, shared by ``fit_full_model.py`` and ``run_recovery_full_model.py``.

``sd_ang`` and ``motiv_ang`` are fixed from the Stimulus Reproduction errors (``derive_fixed_ang_from_estim_errors``);
the four free parameters (``FIT_FREE_PARNAMES``) are fitted to the Sequence Categorization choices with BADS, then
VBMC from the best BADS solution (``fit_full_model_participant``). The recovery simulates reproduction responses
with ``simulate_estim_resp_joint`` and draws generative parameters with ``sample_theta_from_priors``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from motinf.exploratory.model import (
    clip_param,
    df_to_pack,
    fit_metrics,
    fit_model_VBMC,
    get_ll_fast,
    param_config_categ,
    priors_categ,
    run_fit_categ,
    subset_priors,
)


GEN_PARNAMES = ["sd_ang", "sd_inf", "sd_sel", "motiv_ang", "motiv_inf", "motiv_sel"]
FIT_FREE_PARNAMES = ["sd_inf", "sd_sel", "motiv_inf", "motiv_sel"]


def get_fixed_ang_priors() -> dict:
    return subset_priors(priors_categ, GEN_PARNAMES)


def run_fit_categ_bads_then_vbmc(
    df_c,
    model: dict,
    fixedpars: dict,
    priors: dict,
    bads_nit: int,
    *,
    vbmc: bool = True,
) -> dict:
    """BADS fit (``bads_nit`` starts), then, if ``vbmc``, VBMC from the best BADS solution.

    Returns the BADS estimate (``bads_params``, ``bads_ll``, ``bads_aicc``, ``bads_bic``), the VBMC posterior mean and
    SD with the log-likelihood, AICc and BIC at the mean and the ELBO (``vbmc_*``, ``elbo``: empty or NaN without
    VBMC), and the estimate the paper uses (``params``, ``ll``, ``aicc``, ``bic``: VBMC if run, else BADS).
    """
    res_b = run_fit_categ(df_c, model, fixedpars, priors, bads_nit)
    out = {
        "bads_params": res_b["params"],
        "bads_ll": res_b["ll"],
        "bads_aicc": res_b["aicc"],
        "bads_bic": res_b["bic"],
        "vbmc_params": np.array([]),
        "vbmc_params_sd": np.array([]),
        "vbmc_ll": np.nan,
        "vbmc_aicc": np.nan,
        "vbmc_bic": np.nan,
        "elbo": np.nan,
    }
    if not vbmc:
        return {**out, "params": res_b["params"], "ll": res_b["ll"], "aicc": res_b["aicc"], "bic": res_b["bic"]}

    res_v = fit_model_VBMC(
        df_c,
        model["parnames"],
        model["lb"],
        model["ub"],
        plb=model["plb"],
        pub=model["pub"],
        x0=res_b["params"],
        fixedpars=fixedpars,
        priors=priors,
    )
    pnames = list(model["parnames"])
    pv = np.asarray(res_v["params"], dtype=float).ravel()
    psd = np.asarray(res_v["params_sd"], dtype=float).ravel()
    mean_parts = [f"{pn}={float(pv[i]):.4g}" for i, pn in enumerate(pnames) if i < len(pv)]
    print("VBMC posterior mean: " + ", ".join(mean_parts))
    if len(psd) == len(pv) and len(psd) == len(pnames):
        sd_parts = [f"{pn}={float(psd[i]):.4g}" for i, pn in enumerate(pnames)]
        print("VBMC posterior sd:   " + ", ".join(sd_parts))
    print(f"VBMC elbo={float(res_v['elbo']):.4f}")

    ll_v = float(get_ll_fast(pv, model["parnames"], df_to_pack(df_c), fixedpars=fixedpars))
    _, aicc_v, bic_v = fit_metrics(ll_v, len(model["parnames"]), len(df_c))
    out.update(vbmc_params=res_v["params"], vbmc_params_sd=res_v["params_sd"], vbmc_ll=ll_v, vbmc_aicc=aicc_v,
               vbmc_bic=bic_v, elbo=res_v["elbo"])
    return {**out, "params": res_v["params"], "ll": ll_v, "aicc": aicc_v, "bic": bic_v}


def fit_full_model_participant(
    df_c: pd.DataFrame,
    df_e: pd.DataFrame,
    model: dict,
    priors: dict,
    bads_nit: int,
    *,
    vbmc: bool = True,
) -> dict:
    """Full-model fit of one participant: ``sd_ang`` and ``motiv_ang`` fixed from the reproduction trials ``df_e``
    (``fixed``: the output of :func:`derive_fixed_ang_from_estim_errors`), the free parameters of ``model`` fitted to
    the categorization trials ``df_c`` (the keys of :func:`run_fit_categ_bads_then_vbmc`)."""
    fixed = derive_fixed_ang_from_estim_errors(df_e)
    fixedpars = {"sd_ang": float(fixed["sd_ang_categ"]), "motiv_ang": float(fixed["motiv_ang_categ"])}
    return {"fixed": fixed, **run_fit_categ_bads_then_vbmc(df_c, model, fixedpars, priors, bads_nit, vbmc=vbmc)}


def fixed_ang_columns(fixed: dict | None) -> dict:
    """The run-table columns of the fixed ``sd_ang`` / ``motiv_ang`` (NaN and 0 when the fit failed: ``fixed`` None)."""
    f = fixed if fixed is not None else {}
    return {
        "fixed_sd_ang_from_estim": float(f.get("sd_ang_categ", np.nan)),
        "fixed_motiv_ang_from_estim": float(f.get("motiv_ang_categ", np.nan)),
        "sd_base_raw_estim": float(f.get("sd_base_raw", np.nan)),
        "sd_inc_raw_estim_abs": float(f.get("sd_inc_raw", np.nan)),
        "motiv_ang_raw_estim": float(f.get("motiv_ang_raw", np.nan)),
        "n_estim_base": int(f.get("n_base", 0)),
        "n_estim_incentive_abs": int(f.get("n_incentive", 0)),
    }


def param_bounds(name: str) -> tuple[float, float]:
    if name not in param_config_categ:
        raise KeyError(f"Missing bounds for parameter: {name}")
    cfg = param_config_categ[name]
    return float(cfg["lb"]), float(cfg["ub"])


def sample_one_bounded(
    name: str, prior_dist, rng: np.random.Generator, max_tries: int = 200
) -> float:
    lb, ub = param_bounds(name)
    for _ in range(max_tries):
        x = float(prior_dist.rvs(random_state=rng))
        if np.isfinite(x) and lb <= x <= ub:
            return x
    x = float(prior_dist.rvs(random_state=rng))
    return float(np.clip(x, lb, ub))


def sample_theta_from_priors(parnames: list[str], prior_map: dict, rng: np.random.Generator) -> np.ndarray:
    theta = np.empty(len(parnames), dtype=float)
    for i, pn in enumerate(parnames):
        if pn not in prior_map:
            raise KeyError(f"Missing prior for {pn}.")
        theta[i] = sample_one_bounded(pn, prior_map[pn], rng)
    return theta


def theta_to_params(parnames: list[str], theta: np.ndarray) -> dict:
    theta = np.asarray(theta, dtype=float).ravel()
    return {parnames[i]: float(theta[i]) for i in range(len(parnames))}


def _wrap_half_pi(dev: np.ndarray) -> np.ndarray:
    out = dev.copy()
    out[out < -np.pi / 2] = np.pi + out[out < -np.pi / 2]
    out[out > np.pi / 2] = -np.pi + out[out > np.pi / 2]
    return out


def simulate_estim_resp_joint(params: dict, df_e: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Simulated Stimulus Reproduction responses on the trials of ``df_e``: the presented angle plus a Gaussian error
    of SD ``sd_ang`` (neutral) or ``sd_ang * exp(-motiv_ang)`` (gain and loss), wrapped to [-pi/2, pi/2]."""
    angles = df_e["estimang_noisy"].to_numpy(copy=False)
    incentive = df_e["incentive"].to_numpy(copy=False)
    sd = clip_param(params["sd_ang"], "sd_ang", param_config_categ)
    sd_inc = clip_param(sd * np.exp(-params["motiv_ang"]), "sd_ang", param_config_categ)
    dev = rng.normal(loc=0.0, scale=np.where(np.abs(incentive) == 1, sd_inc, sd), size=len(angles))
    return (angles % np.pi + _wrap_half_pi(dev)) % np.pi


def _circular_sd_half_pi(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    c = np.abs(np.mean(np.exp(1j * x)))
    c = np.clip(c, 1e-12, 1.0)
    return float(np.sqrt(-2.0 * np.log(c)))


def derive_fixed_ang_from_estim_errors(
    df_e: pd.DataFrame,
    *,
    resp_col: str = "resp_estim",
    angle_col: str = "estimang_noisy",
    incentive_col: str = "incentive",
    eps_sd: float = 1e-6,
) -> dict:
    if resp_col not in df_e or angle_col not in df_e:
        raise KeyError(f"df_e must contain {resp_col!r} and {angle_col!r}.")
    if incentive_col not in df_e:
        raise KeyError(f"df_e must contain {incentive_col!r}.")

    resp = pd.to_numeric(df_e[resp_col], errors="coerce").to_numpy(dtype=float)
    ang = pd.to_numeric(df_e[angle_col], errors="coerce").to_numpy(dtype=float)
    inc = pd.to_numeric(df_e[incentive_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(resp) & np.isfinite(ang) & np.isfinite(inc)
    if not np.any(valid):
        raise ValueError("No valid estimation rows to derive fixed sd_ang/motiv_ang.")

    dev = _wrap_half_pi((resp[valid] - ang[valid]).astype(float))
    inc_v = inc[valid]
    base_mask = inc_v == 0
    inc_mask = np.abs(inc_v) == 1
    if not np.any(base_mask):
        raise ValueError("No neutral (incentive == 0) estimation trials available.")
    if not np.any(inc_mask):
        raise ValueError("No incentivized (abs(incentive) == 1) estimation trials available.")

    sd_base_raw = max(_circular_sd_half_pi(dev[base_mask]), eps_sd)
    sd_inc_raw = max(_circular_sd_half_pi(dev[inc_mask]), eps_sd)
    motiv_raw = float(np.log(sd_base_raw / sd_inc_raw))

    sd_ang_categ = float(clip_param(sd_base_raw, "sd_ang", param_config_categ))
    motiv_ang_categ = float(clip_param(motiv_raw, "motiv_ang", param_config_categ))

    return {
        "sd_base_raw": float(sd_base_raw),
        "sd_inc_raw": float(sd_inc_raw),
        "motiv_ang_raw": float(motiv_raw),
        "sd_ang_categ": sd_ang_categ,
        "motiv_ang_categ": motiv_ang_categ,
        "n_base": int(np.sum(base_mask)),
        "n_incentive": int(np.sum(inc_mask)),
    }


def derive_ang_params_from_estim_errors(
    df_e: pd.DataFrame,
    *,
    resp_col: str = "resp_estim",
    angle_col: str = "estimang_noisy",
    incentive_col: str = "incentive",
    eps_sd: float = 1e-6,
) -> dict[str, float]:
    """Neutral ``sd_ang`` plus gain/loss ``motiv_ang`` from estimation errors.

    Uses the same circular-SD logic as :func:`derive_fixed_ang_from_estim_errors`
    (``sd_ang`` matches ``sd_ang_categ`` / full-model ``fixed_sd_ang``).
    """
    if resp_col not in df_e or angle_col not in df_e:
        raise KeyError(f"df_e must contain {resp_col!r} and {angle_col!r}.")
    if incentive_col not in df_e:
        raise KeyError(f"df_e must contain {incentive_col!r}.")

    resp = pd.to_numeric(df_e[resp_col], errors="coerce").to_numpy(dtype=float)
    ang = pd.to_numeric(df_e[angle_col], errors="coerce").to_numpy(dtype=float)
    inc = pd.to_numeric(df_e[incentive_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(resp) & np.isfinite(ang) & np.isfinite(inc)
    if not np.any(valid):
        raise ValueError("No valid estimation rows to derive angular parameters.")

    dev = _wrap_half_pi((resp[valid] - ang[valid]).astype(float))
    inc_v = inc[valid]
    base_mask = inc_v == 0
    gain_mask = inc_v == 1
    loss_mask = inc_v == -1
    if not np.any(base_mask):
        raise ValueError("No neutral (incentive == 0) estimation trials available.")
    if not np.any(gain_mask):
        raise ValueError("No gain (incentive == 1) estimation trials available.")
    if not np.any(loss_mask):
        raise ValueError("No loss (incentive == -1) estimation trials available.")

    sd_base_raw = max(_circular_sd_half_pi(dev[base_mask]), eps_sd)
    sd_gain_raw = max(_circular_sd_half_pi(dev[gain_mask]), eps_sd)
    sd_loss_raw = max(_circular_sd_half_pi(dev[loss_mask]), eps_sd)

    sd_ang = float(clip_param(sd_base_raw, "sd_ang", param_config_categ))
    motiv_ang_g = float(
        clip_param(np.log(sd_base_raw / sd_gain_raw), "motiv_ang", param_config_categ)
    )
    motiv_ang_l = float(
        clip_param(np.log(sd_base_raw / sd_loss_raw), "motiv_ang", param_config_categ)
    )
    return {
        "sd_ang": sd_ang,
        "motiv_ang_g": motiv_ang_g,
        "motiv_ang_l": motiv_ang_l,
    }


def build_estimation_ang_params_table(df_est: pd.DataFrame) -> pd.DataFrame:
    """Participant-level ``sd_ang``, ``motiv_ang_g``, ``motiv_ang_l`` from estimation trials."""
    required = {"participant", "resp_estim", "estimang_noisy", "incentive"}
    missing = required.difference(df_est.columns)
    if missing:
        raise KeyError(f"Estimation table missing required columns: {sorted(missing)}")

    d = df_est[list(required)].copy()
    d["participant"] = d["participant"].astype(str)
    d["incentive"] = pd.to_numeric(d["incentive"], errors="coerce")
    d = d[d["incentive"].isin([-1.0, 0.0, 1.0])]
    counts = d.groupby("participant", observed=True)["incentive"].nunique(dropna=True)
    keep = counts[counts == 3].index

    rows: list[dict[str, float | str]] = []
    for pid in keep:
        g = d[d["participant"] == pid]
        try:
            params = derive_ang_params_from_estim_errors(g)
        except (ValueError, KeyError):
            continue
        rows.append({"participant": str(pid), **params})

    if not rows:
        raise RuntimeError("No usable participant angular parameters from estimation data.")
    return pd.DataFrame(rows).drop_duplicates(subset=["participant"], keep="first").reset_index(drop=True)


def correlation(x: np.ndarray, y: np.ndarray, method: str = "pearson") -> float:
    if len(x) < 2:
        return float("nan")
    sx = pd.Series(x)
    sy = pd.Series(y)
    return float(sx.corr(sy, method=method))
