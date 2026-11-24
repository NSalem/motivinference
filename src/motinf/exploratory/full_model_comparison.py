"""Full-model fit tables and group BMC of the full model against inference-only models.

``load_full_model_fit_bundle`` reads ``fit_full_model.npz`` (the one reader of that file; ``usable_fits`` says which
participants have a usable fit) and ``full_model_fit_table`` makes it a tidy table, including the whole-data AICc at
the best BADS solution and at the VBMC posterior mean; ``noise_per_cond_global_aicc_table`` scores the
pre-registered per-condition single-noise inference fit on the whole dataset; ``group_bmc_from_aicc`` runs
random-effects group BMC (``groupBMC``) with ``-0.5 * AICc`` as log evidence. Used by
``scripts/exploratory/stats/run_full_vs_inference_only_bmc.py`` (Results) and, through
:func:`motinf.exploratory.full_model_results.load_full_model_params`, by the full-model statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from motinf.exploratory.full_model_fitting import FIT_FREE_PARNAMES
from motinf.exploratory.stats import group_bmc


@dataclass(frozen=True)
class FullModelFitBundle:
    """Per-participant real full-model fit, one row per participant (segment-local ids)."""

    participants: np.ndarray  # (n,) str
    sd_ang: np.ndarray  # (n,) fixed from Estimation
    motiv_ang: np.ndarray  # (n,) fixed from Estimation
    theta_point: np.ndarray  # (n, 4) best point estimate, columns == FIT_FREE_PARNAMES
    theta_bads: np.ndarray  # (n, 4) BADS-only estimate (pre-VBMC)
    aicc_bads: np.ndarray  # (n,) whole-data AICc of the BADS-only full-model fit
    ll_bads: np.ndarray  # (n,)
    aicc_vbmc: np.ndarray  # (n,) whole-data AICc at the VBMC posterior mean (NaN where VBMC was not run)
    ll_vbmc: np.ndarray  # (n,)
    point_source: np.ndarray  # (n,) 'vbmc' or 'bads'


def load_full_model_fit_bundle(npz_path: Path) -> FullModelFitBundle:
    """Load one segment's ``fit_full_model.npz`` (see ``fit_full_model.py`` / ``full_model_cohort.py``).

    The point estimate is the VBMC posterior mean when the fit ran VBMC after BADS and the mean is finite, the best
    BADS solution otherwise."""
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing full-model NPZ: {npz_path}")
    fit = np.load(npz_path, allow_pickle=True)
    parnames = [str(x) for x in np.asarray(fit["fit_free_parnames"], dtype=object).tolist()]
    if parnames != FIT_FREE_PARNAMES:
        raise ValueError(f"Unexpected fit_free_parnames in {npz_path}: {parnames} (expected {FIT_FREE_PARNAMES})")

    participants = np.asarray(fit["participants"]).astype(str)
    n = len(participants)
    use_vbmc = bool(np.asarray(fit["vbmc_after_bads"]).item())
    pars_bads = np.asarray(fit["pars_bads"], dtype=object)
    pars_vbmc = np.asarray(fit["pars_vbmc"], dtype=object) if "pars_vbmc" in fit.files else None
    ll_bads = np.asarray(fit["ll_bads"], dtype=float)
    aicc_bads = np.asarray(fit["aicc_bads"], dtype=float)
    nan_n = np.full(n, np.nan, dtype=float)
    aicc_vbmc = np.asarray(fit["aicc_vbmc_mean"], dtype=float) if "aicc_vbmc_mean" in fit.files else nan_n
    ll_vbmc = np.asarray(fit["ll_vbmc_mean"], dtype=float) if "ll_vbmc_mean" in fit.files else nan_n
    fixed_sd_ang = np.asarray(fit["fixed_sd_ang"], dtype=float)
    fixed_motiv_ang = np.asarray(fit["fixed_motiv_ang"], dtype=float)

    theta_bads = np.full((n, len(FIT_FREE_PARNAMES)), np.nan, dtype=float)
    theta_point = np.full((n, len(FIT_FREE_PARNAMES)), np.nan, dtype=float)
    point_source = np.full(n, "", dtype=object)

    for i in range(n):
        cand_bads = np.asarray(pars_bads[i], dtype=float).ravel()
        if cand_bads.size == len(FIT_FREE_PARNAMES):
            theta_bads[i] = cand_bads
        theta_i = None
        src = ""
        if use_vbmc and pars_vbmc is not None:
            cand_v = np.asarray(pars_vbmc[i], dtype=float).ravel()
            if cand_v.size == len(FIT_FREE_PARNAMES) and np.all(np.isfinite(cand_v)):
                theta_i, src = cand_v, "vbmc"
        if theta_i is None and cand_bads.size == len(FIT_FREE_PARNAMES) and np.all(np.isfinite(cand_bads)):
            theta_i, src = cand_bads, "bads"
        if theta_i is not None:
            theta_point[i] = theta_i
            point_source[i] = src

    return FullModelFitBundle(
        participants=participants,
        sd_ang=fixed_sd_ang,
        motiv_ang=fixed_motiv_ang,
        theta_point=theta_point,
        theta_bads=theta_bads,
        aicc_bads=aicc_bads,
        ll_bads=ll_bads,
        aicc_vbmc=aicc_vbmc,
        ll_vbmc=ll_vbmc,
        point_source=point_source,
    )


def usable_fits(b: FullModelFitBundle) -> np.ndarray:
    """Participants with a usable fit: finite point estimate, finite fixed ``sd_ang`` and ``motiv_ang``, and a finite
    log-likelihood at the point estimate (of VBMC or BADS, whichever gave it)."""
    ll_point = np.where(b.point_source == "vbmc", b.ll_vbmc, b.ll_bads)
    return (np.all(np.isfinite(b.theta_point), axis=1) & np.isfinite(b.sd_ang) & np.isfinite(b.motiv_ang)
            & np.isfinite(ll_point))


def full_model_fit_table(root: Path, seg: str) -> pd.DataFrame:
    """One segment's full-model fit as a tidy DataFrame (unprefixed participant ids; :func:`usable_fits` only)."""
    npz_path = root / "results" / "exploratory" / seg / "modeling" / "fit_files" / "fit_full_model.npz"
    b = load_full_model_fit_bundle(npz_path)
    rows: list[dict[str, Any]] = []
    for i in np.flatnonzero(usable_fits(b)):
        row = {
            "participant": str(b.participants[i]),
            "segment": seg,
            "sd_ang": float(b.sd_ang[i]),
            "motiv_ang": float(b.motiv_ang[i]),
            "aicc_bads_full": float(b.aicc_bads[i]),
            "ll_bads_full": float(b.ll_bads[i]),
            "aicc_vbmc_full": float(b.aicc_vbmc[i]),
            "ll_vbmc_full": float(b.ll_vbmc[i]),
            "point_source": str(b.point_source[i]),
        }
        for j, pn in enumerate(FIT_FREE_PARNAMES):
            row[pn] = float(b.theta_point[i, j])
            row[f"{pn}_bads"] = float(b.theta_bads[i, j]) if np.isfinite(b.theta_bads[i, j]) else np.nan
        rows.append(row)
    if not rows:
        raise RuntimeError(f"No usable participant rows in {npz_path}")
    return pd.DataFrame(rows)


def noise_per_cond_global_aicc_table(root: Path, seg: str) -> pd.DataFrame:
    """Global (whole-dataset) AICc/BIC of the per-condition inference-noise fit in ``fit_noise_per_cond.npz``.

    ``fit_noise_per_cond.npz`` (``scripts/prereg/fit_noise_per_cond.py``) fits one free noise parameter (here
    ``sd_inf``; no other channel, no incentive-modulation link between conditions) *separately per incentive
    condition* (3 single-parameter fits per participant, each on that condition's own trial subset), and stores each
    condition's own log-likelihood/AICc/BIC. This treats the 3 independently-fit conditions as one unconstrained
    "noise-per-condition" model (3 free parameters total, no shared structure) and sums log-likelihoods (valid since
    the three conditions are disjoint trial sets) to get one global AICc/BIC comparable to the full model's
    whole-dataset AICc -- the least-constrained ("saturated") alternative to the full model's parametric
    ``sd_X_v = sd_X * exp(-motiv_X)`` link across conditions.
    """
    npz_path = root / "results" / "prereg" / seg / "model_fit" / "fit_noise_per_cond.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing noise-per-condition NPZ: {npz_path}")
    d = np.load(npz_path, allow_pickle=True)
    participants = np.asarray(d["participants"]).astype(str)
    incentives = np.asarray(d["incentives"], dtype=float)
    ll = np.asarray(d["ll_inf"], dtype=float)  # (n_p, n_i)
    k_total = int(ll.shape[1])  # one free parameter per condition

    inf_csv = root / "data" / "processed" / seg / "data_clean_inf.csv"
    if not inf_csv.exists():
        raise FileNotFoundError(f"Missing trial data for global AICc denominator: {inf_csv}")
    df_inf = pd.read_csv(inf_csv, usecols=["participant", "incentive"])
    df_inf["participant"] = df_inf["participant"].astype(str)
    df_inf["incentive"] = df_inf["incentive"].astype(float)
    n_trials_by_pid_inc = df_inf.groupby(["participant", "incentive"]).size()

    rows: list[dict[str, Any]] = []
    for ip, pid in enumerate(participants):
        ll_row = ll[ip]
        if np.any(~np.isfinite(ll_row)):
            continue
        try:
            n_total = int(sum(n_trials_by_pid_inc.loc[(str(pid), float(inc))] for inc in incentives))
        except KeyError:
            continue
        ll_total = float(np.sum(ll_row))
        aic = 2 * k_total - 2 * ll_total
        denom = n_total - k_total - 1
        aicc = aic + (2 * k_total * (k_total + 1)) / denom if denom > 0 else np.nan
        bic = -2 * ll_total + k_total * np.log(n_total) if n_total > 0 else np.nan
        rows.append(
            {
                "participant": str(pid),
                "segment": seg,
                "ll_noise_per_cond_inf": ll_total,
                "aicc_noise_per_cond_inf": float(aicc),
                "bic_noise_per_cond_inf": float(bic),
                "k_noise_per_cond": k_total,
                "n_trials_noise_per_cond": n_total,
            }
        )
    if not rows:
        raise RuntimeError(f"No usable participant rows in {npz_path}")
    return pd.DataFrame(rows)


def group_bmc_from_log_evidence(log_ev_by_model: dict[str, np.ndarray], model_names: list[str]) -> pd.DataFrame:
    """Random-effects group BMC (protected exceedance probability) from per-participant log-evidence.

    ``log_ev_by_model[m]`` is a (n_participants,) array of log-model-evidence (or a valid proxy:
    summed cross-validated held-out log-likelihood, or ``-0.5 * AICc``) for model ``m``, fed
    directly to ``groupBMC`` (:func:`motinf.exploratory.stats.group_bmc`). Complete cases only (drops participants
    with a non-finite value for any compared model).
    """
    mat = np.stack([np.asarray(log_ev_by_model[m], dtype=float) for m in model_names], axis=0)
    keep = np.all(np.isfinite(mat), axis=0)
    n_total = mat.shape[1]
    mat = mat[:, keep]
    n_kept = mat.shape[1]
    if n_kept < 3:
        return pd.DataFrame(
            {
                "model": model_names,
                "message": [
                    f"Too few complete participants for GroupBMC (need >=3, got {n_kept} of {n_total})."
                ]
                * len(model_names),
            }
        )
    result = group_bmc(mat)
    return pd.DataFrame(
        {
            "model": model_names,
            "n_participants": n_kept,
            "n_participants_total": n_total,
            "frequency_mean": result["frequency_mean"],
            "exceedance_probability": result["exceedance_probability"],
            "protected_exceedance_probability": result["protected_exceedance_probability"],
        }
    )


def group_bmc_from_aicc(aicc_by_model: dict[str, np.ndarray], model_names: list[str]) -> pd.DataFrame:
    """Group BMC from per-participant AICc (``-0.5 * AICc`` as the log-evidence proxy).

    Same approximation as ``single_modulation_fits`` and the model recoveries (``identifiability``).
    """
    log_ev = {m: -0.5 * np.asarray(aicc_by_model[m], dtype=float) for m in model_names}
    return group_bmc_from_log_evidence(log_ev, model_names)
