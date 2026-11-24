"""Cohort and the layout of ``fit_full_model.npz`` for ``fit_full_model.py``.

The cohort is the participants with trials in both tasks, in the order of the Sequence Categorization file. The fit
file holds one slot per cohort participant (``SLOT_KEYS``) and the settings of the run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from motinf.exploratory.cohorts import apply_row_limits
from motinf.exploratory.full_model_fitting import FIT_FREE_PARNAMES

# One value per cohort participant; the parameter vectors are object arrays (empty without VBMC).
SLOT_KEYS = (
    "pars_bads",
    "ll_bads",
    "aicc_bads",
    "bic_bads",
    "pars_vbmc",
    "pars_vbmc_sd",
    "ll_vbmc_mean",
    "aicc_vbmc_mean",
    "bic_vbmc_mean",
    "elbo_vbmc",
    "fixed_sd_ang",
    "fixed_motiv_ang",
)
OBJECT_SLOT_KEYS = ("pars_bads", "pars_vbmc", "pars_vbmc_sd")


def load_clean_dataframes(
    root: Path,
    exp_name: str,
    max_ntrials: int,
    max_nsess: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_inf_path = root / "data" / "processed" / exp_name / "data_clean_inf.csv"
    data_est_path = root / "data" / "processed" / exp_name / "data_clean_est.csv"
    if not data_inf_path.exists() or not data_est_path.exists():
        raise FileNotFoundError(f"Missing data files under {root / 'data' / 'processed' / exp_name}")
    df_inf = pd.read_csv(str(data_inf_path))
    df_inf["seqlen"] = df_inf["seqlen"].astype(int)
    df_inf = apply_row_limits(df_inf, max_ntrials, max_nsess)
    df_est = pd.read_csv(str(data_est_path))
    df_est = apply_row_limits(df_est, max_ntrials, max_nsess)
    return df_inf, df_est


def ordered_cohort_participants(
    df_inf: pd.DataFrame,
    df_est: pd.DataFrame,
    max_nsubs: int,
) -> np.ndarray:
    """Intersection of participants, preserving order from inference CSV."""
    part_inf = np.asarray(pd.unique(df_inf["participant"]), dtype=object)
    part_est_set = set(pd.unique(df_est["participant"]))
    cohort = np.array([p for p in part_inf if p in part_est_set], dtype=object)
    if max_nsubs > 0:
        cohort = cohort[:max_nsubs]
    if len(cohort) == 0:
        raise ValueError("No participants available after filtering.")
    return cohort


def allocate_slots(n: int) -> dict[str, Any]:
    """Empty slots (``SLOT_KEYS``) for ``n`` participants: NaN, NaN parameter vectors for BADS, empty for VBMC."""
    out: dict[str, Any] = {}
    for key in SLOT_KEYS:
        if key in OBJECT_SLOT_KEYS:
            out[key] = np.empty(n, dtype=object)
            for i in range(n):
                out[key][i] = np.full(len(FIT_FREE_PARNAMES), np.nan) if key == "pars_bads" else np.array([])
        else:
            out[key] = np.full(n, np.nan)
    return out


def fit_file_dict(
    slots: dict[str, Any],
    *,
    cohort_participants: np.ndarray,
    vbmc_after_bads: bool,
    use_priors_fit: bool,
    bads_nit: int,
    seed: int,
    max_nsubs: int,
    max_ntrials: int,
    max_nsess: int,
) -> dict[str, Any]:
    """Keyword arguments for ``np.savez``: the slots, the cohort and the settings of the run."""
    participants = np.asarray(cohort_participants, dtype=object)
    return {
        "participants": participants,
        "fit_free_parnames": np.array(FIT_FREE_PARNAMES, dtype=object),
        **{key: slots[key] for key in SLOT_KEYS},
        "vbmc_after_bads": bool(vbmc_after_bads),
        "use_priors_fit": bool(use_priors_fit),
        "bads_nit": int(bads_nit),
        "seed": int(seed),
        "cohort_n": int(len(participants)),
        "cohort_participants": participants,
        "max_nsubs": int(max_nsubs),
        "max_ntrials": int(max_ntrials),
        "max_nsess": int(max_nsess),
    }
