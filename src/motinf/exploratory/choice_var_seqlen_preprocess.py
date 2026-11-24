"""Observed against predicted choice variability over sequence length: merge and participant exclusion.

Behaviour and model predictions are inner-merged per participant x incentive x sequence length, and a participant
is excluded when any of their cells lies more than ``CHOICE_VAR_SEQLEN_MAD_THRESHOLD`` (5) MAD from the median of
that cell (two-sided, ``scale="normal"``): the choice-variability exclusion used throughout the paper (Figs 2d, 3c, d,
Supp. Figs 4c, d and 6, the natural-scale rmANOVA of Supp. Table 1). :func:`var_red_natural_long_by_seqlen` gives
the incentive effect (neutral − |incentive| = 1) per sequence length.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

CHOICE_VAR_SEQLEN_MAD_THRESHOLD: float = 5

# Reference sequence length for OLS ``DV ~ seqlen`` in exploratory slope / intercept fits: the fitted **intercept** is
# ``E[y]`` at this ``seqlen``; the **slope** (change in ``y`` per unit ``seqlen``) is unchanged from an uncentered fit.
CHOICE_VAR_SEQLEN_CENTER_FOR_SLOPE: float = 8.0


def mad_filter_participants_if_any_cell_mad_outlier(
    df: pd.DataFrame,
    *,
    participant_col: str,
    beh_col: str,
    mad_threshold: float = CHOICE_VAR_SEQLEN_MAD_THRESHOLD,
) -> tuple[pd.DataFrame, int, int]:
    """Drop every participant with a value of ``beh_col`` more than ``mad_threshold`` MAD from the median of its
    (incentive, seqlen) cell, ``|x - median| > k * MAD`` (a cell with fewer than 3 finite values, or a zero MAD, flags
    nobody). Returns the rows kept, the number of rows removed and the number of participants removed."""
    n_before = len(df)
    k = float(mad_threshold)
    flagged: set[str] = set()

    for _, g in df.groupby(["incentive", "seqlen"], sort=False):
        v = g[beh_col].to_numpy(dtype=float, copy=False)
        pids = g[participant_col].astype(str).to_numpy()
        mask = np.isfinite(v)
        if int(mask.sum()) < 3:
            continue
        vv = v[mask]
        pp = pids[mask]
        med = float(np.median(vv))
        mad = float(median_abs_deviation(vv, scale="normal", nan_policy="omit"))
        if not np.isfinite(mad) or mad <= 0:
            continue
        flagged.update(str(pid) for pid in pp[np.abs(vv - med) > k * mad])

    if not flagged:
        return df.copy(), 0, 0
    dfc = df[~df[participant_col].astype(str).isin(flagged)].copy()
    return dfc, n_before - len(dfc), len(flagged)


def merge_and_preprocess_choice_var_beh_pred(
    df_beh_seq: pd.DataFrame,
    df_pred_per: pd.DataFrame,
    *,
    report_label: str | None = None,
) -> pd.DataFrame:
    """Behaviour (``choice_var``) inner-merged with the predictions (``pred_choice_var``) per participant x incentive x
    sequence length, cells with a missing value dropped, then the participants with an extreme cell excluded
    (:func:`mad_filter_participants_if_any_cell_mad_outlier`). ``report_label``: print the counts under that label.
    An empty frame when a required column is missing."""
    keys = ["participant", "incentive", "seqlen"]
    if not {*keys, "choice_var"}.issubset(df_beh_seq.columns) or not {*keys, "pred_choice_var"}.issubset(
        df_pred_per.columns
    ):
        return pd.DataFrame()
    m = df_beh_seq.merge(df_pred_per, on=keys, how="inner")
    m = m.dropna(subset=["incentive", "seqlen", "choice_var", "pred_choice_var"]).copy()
    n_cells = len(m)
    m, n_rows, n_participants = mad_filter_participants_if_any_cell_mad_outlier(
        m, participant_col="participant", beh_col="choice_var"
    )
    if report_label is not None:
        print(
            f"[choice_var vs seqlen] {report_label}: {n_cells} participant x incentive x seqlen cells; "
            f"{n_participants} participants ({n_rows} cells) excluded, with a cell beyond "
            f"{CHOICE_VAR_SEQLEN_MAD_THRESHOLD:g} MAD of its median"
        )
    return m


def var_red_natural_long_by_seqlen(
    df: pd.DataFrame,
    vcol: str,
) -> pd.DataFrame:
    """Per (participant, seqlen): mean(V | |inc|=0) − mean(V | |inc|=1).

    Pools signed incentives −1 and +1 into |inc|=1 before cell means.
    """
    need = {"participant", "incentive", "seqlen", vcol}
    if not need.issubset(df.columns):
        return pd.DataFrame(columns=["participant", "seqlen", "var_red"])
    d = df.copy()
    d["participant"] = d["participant"].astype(str)
    d["incentive_abs"] = np.abs(d["incentive"].astype(float)).round().astype(int)
    means = d.groupby(["participant", "seqlen", "incentive_abs"], as_index=False)[vcol].mean()
    wide = means.pivot(index=["participant", "seqlen"], columns="incentive_abs", values=vcol)
    if 0 not in wide.columns or 1 not in wide.columns:
        return pd.DataFrame(columns=["participant", "seqlen", "var_red"])
    v0 = wide[0].astype(float)
    v1 = wide[1].astype(float)
    out = wide.reset_index()[["participant", "seqlen"]].copy()
    out["var_red"] = (v0 - v1).values
    return out
