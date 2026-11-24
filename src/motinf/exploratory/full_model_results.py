"""Full-model helpers: parameter tables, channel variances, predictions and knockouts.

Used by ``scripts/exploratory/stats/run_full_model_stats.py`` (Figure 3, Supp. Figure 6, Supp. Table 4).
The full model fixes the estimation noise (``sd_ang``) and its incentive modulation (``motiv_ang``) from
the Estimation task and fits ``sd_inf, sd_sel, motiv_inf, motiv_sel``. Notation:

    sigma_X  ->  sd_X      (X in {ang, inf, sel})
    delta_X  ->  motiv_X
    sigma_X_v = sd_X * exp(-motiv_X)   (effective noise under |incentive| = 1)
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from motinf.exploratory.cohorts import participant_prefixed
from motinf.exploratory.full_model_comparison import load_full_model_fit_bundle, usable_fits
from motinf.exploratory.full_model_fitting import FIT_FREE_PARNAMES
from motinf.exploratory.model import df_to_pack, get_ppost_fast


CHANNELS: tuple[str, ...] = ("ang", "inf", "sel")
ALL_PAR_NAMES: tuple[str, ...] = (
    "sd_ang",
    "sd_inf",
    "sd_sel",
    "motiv_ang",
    "motiv_inf",
    "motiv_sel",
)


def add_effective_and_variance(df_pars: pd.DataFrame, mean_seqlen: float) -> pd.DataFrame:
    """Append derived `sd_X_v`, `var_X`, `var_X_v`, `total_var_X(_v)` and totals.

    Channel weights follow `get_ppost_fast`'s contribution to choice variance:
    `sigma_post^2 = mu_llr_term(sd_ang) + (seqlen-1)*sd_inf^2 + sd_sel^2`.
    For the variance partition each channel is weighted by its number of contributions in a
    sequence of mean length: `sd_ang^2` by `mean_seqlen`, `sd_inf^2` by `mean_seqlen-1` and
    `sd_sel^2` by `1`.
    """
    out = df_pars.copy()
    weights = {
        "ang": float(mean_seqlen),
        "inf": float(mean_seqlen) - 1.0,
        "sel": 1.0,
    }
    for ch in CHANNELS:
        sd = out[f"sd_{ch}"].astype(float)
        motiv = out[f"motiv_{ch}"].astype(float)
        sd_v = sd * np.exp(-motiv)
        out[f"sd_{ch}_v"] = sd_v
        out[f"var_{ch}"] = sd**2
        out[f"var_{ch}_v"] = sd_v**2
        out[f"total_var_{ch}"] = (sd**2) * weights[ch]
        out[f"total_var_{ch}_v"] = (sd_v**2) * weights[ch]
    out["total_var"] = out[[f"total_var_{ch}" for ch in CHANNELS]].sum(axis=1)
    out["total_var_v"] = out[[f"total_var_{ch}_v" for ch in CHANNELS]].sum(axis=1)
    return out


def compute_motiv_correlation_table(
    df_pars: pd.DataFrame,
    cols: tuple[str, ...] = ("motiv_ang", "motiv_inf", "motiv_sel"),
) -> pd.DataFrame:
    """Pairwise Pearson/Spearman correlations among full-model incentive effects.

    Returns a long-form table with one row per unordered pair of motivation parameters.
    """
    missing = [c for c in cols if c not in df_pars.columns]
    if missing:
        raise KeyError(f"Missing incentive-effect columns: {missing}")

    d = df_pars[list(cols)].dropna(how="any").copy()
    rows: list[dict] = []
    for x, y in combinations(cols, 2):
        if d.empty:
            rows.append(
                {
                    "x": x,
                    "y": y,
                    "n": 0,
                    "pearson_r": np.nan,
                    "pearson_p": np.nan,
                    "spearman_r": np.nan,
                    "spearman_p": np.nan,
                }
            )
            continue
        dx = d[x].to_numpy(dtype=float)
        dy = d[y].to_numpy(dtype=float)
        pr, pp = pearsonr(dx, dy)
        sr, sp = spearmanr(dx, dy)
        rows.append(
            {
                "x": x,
                "y": y,
                "n": int(len(dx)),
                "pearson_r": float(pr),
                "pearson_p": float(pp),
                "spearman_r": float(sr),
                "spearman_p": float(sp),
            }
        )
    return pd.DataFrame(rows)


def _predict_subject(pars_full: dict, df_sub: pd.DataFrame) -> pd.DataFrame:
    """Trial-level model predictions for one subject using `get_ppost_fast`."""
    data = df_to_pack(df_sub)
    ppost, xpost, spost = get_ppost_fast(pars_full, data, kappa=0.5, sigstr=1)
    out = df_sub[["participant", "incentive", "seqlen"]].copy().reset_index(drop=True)
    out["ppost"] = ppost
    out["xpost"] = xpost
    out["spost"] = spost
    out["choicevar"] = spost**2
    return out


def compute_model_predictions(
    df_inf: pd.DataFrame,
    df_pars: pd.DataFrame,
    knockout_motivs: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Trial-level model predictions per participant with optional knockout.

    `knockout_motivs` is a subset of `("ang", "inf", "sel")`. The corresponding
    `motiv_X` is set to 0 before generating predictions, eliminating the
    incentive effect on that channel while keeping baseline `sd_X` intact.
    """
    bad = [m for m in knockout_motivs if m not in CHANNELS]
    if bad:
        raise ValueError(f"Unknown knockout channel(s): {bad}; expected subset of {CHANNELS}.")

    parts_pars = df_pars.set_index("participant")
    parts = parts_pars.index.to_list()

    df_inf = df_inf.copy()
    df_inf["participant"] = df_inf["participant"].astype(str)
    df_inf["seqlen"] = df_inf["seqlen"].astype(int)
    df_inf["incentive"] = df_inf["incentive"].astype(float)

    chunks: list[pd.DataFrame] = []
    for pid in parts:
        df_sub = df_inf[df_inf["participant"] == pid]
        if df_sub.empty:
            continue
        prow = parts_pars.loc[pid]
        pars: dict = {pn: float(prow[pn]) for pn in ALL_PAR_NAMES}
        for ch in knockout_motivs:
            pars[f"motiv_{ch}"] = 0.0
        chunks.append(_predict_subject(pars, df_sub))

    if not chunks:
        raise RuntimeError("No predictions generated; check that participants overlap.")
    return pd.concat(chunks, ignore_index=True)


def aggregate_predictions_per_seqlen(df_pred: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trial-level predictions to per-participant×seqlen×incentive means (``pred_choice_var``)."""
    return (
        df_pred.groupby(["participant", "incentive", "seqlen"], as_index=False)
        .agg(pred_choice_var=("choicevar", "mean"))
        .sort_values(["participant", "incentive", "seqlen"])
        .reset_index(drop=True)
    )


def knockout_model_specs() -> list[dict]:
    """The models of the knockout analysis (Figure 3c, d): the full model and the three models without one incentive
    modulation. ``id``: model label in the outputs; ``knockouts``: the channels whose ``motiv_X`` is set to 0."""
    return [{"id": "full", "knockouts": ()}] + [{"id": f"ko_{ch}", "knockouts": (ch,)} for ch in CHANNELS]


def load_full_model_params(npz_path: Path) -> pd.DataFrame:
    """Per participant, the fixed estimation parameters (``sd_ang``, ``motiv_ang``) and the fitted
    ``sd_inf, sd_sel, motiv_inf, motiv_sel`` of one ``fit_full_model.npz`` (VBMC posterior mean when the fit
    ran VBMC after BADS and it is finite, the BADS solution otherwise); participants without a usable fit
    (:func:`motinf.exploratory.full_model_comparison.usable_fits`) are left out."""
    b = load_full_model_fit_bundle(npz_path)
    rows: list[dict] = []
    for i in np.flatnonzero(usable_fits(b)):
        row = {
            "participant": str(b.participants[i]),
            "sd_ang": float(b.sd_ang[i]),
            "motiv_ang": float(b.motiv_ang[i]),
        }
        for j, pn in enumerate(FIT_FREE_PARNAMES):
            row[pn] = float(b.theta_point[i, j])
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No usable participant rows found in {npz_path}")
    return pd.DataFrame(rows).reset_index(drop=True)


def load_full_model_params_for_segments(root: Path, segments: list[str]) -> pd.DataFrame:
    """``load_full_model_params`` of each segment's ``fit_full_model.npz``, concatenated (participant ids
    prefixed with the segment when there are several)."""
    prefix_pids = len(segments) > 1
    chunks: list[pd.DataFrame] = []
    for seg in segments:
        npz_path = (
            root
            / "results"
            / "exploratory"
            / seg
            / "modeling"
            / "fit_files"
            / "fit_full_model.npz"
        )
        df = load_full_model_params(npz_path)
        if prefix_pids:
            df["participant"] = df["participant"].astype(str).map(
                lambda p, s=seg: participant_prefixed(s, p)
            )
        chunks.append(df)
    out = pd.concat(chunks, ignore_index=True)
    if out.empty:
        raise RuntimeError(
            "No usable participant rows after loading fit_full_model.npz for segments: "
            + ", ".join(segments)
        )
    return out
