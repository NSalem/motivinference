"""Statistics helpers shared by the stats and tables stages: Holm correction, dz with its exact confidence interval,
group Bayesian model selection, consistency of choices to repeated sequences, and per-participant slopes / intercepts
of choice variability over sequence length."""

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import nct
from statsmodels import api as sm

from motinf.exploratory.choice_var_seqlen_preprocess import CHOICE_VAR_SEQLEN_CENTER_FOR_SLOPE


def holm(ps: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the order given; a NaN stays NaN and does not count as a test."""
    ps = np.asarray(ps, dtype=float)
    adj = np.full(ps.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(ps))
    m = len(finite)
    running = 0.0
    for rank, idx in enumerate(finite[np.argsort(ps[finite])]):
        running = max(running, (m - rank) * ps[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


def dz_ci(t: float, n: int, conf: float = 0.95) -> tuple[float, float, float]:
    """dz = t / sqrt(n) with an exact CI by inverting the noncentral t (narrow bracket
    around t avoids the nan scipy returns for very large |ncp|)."""
    df = n - 1
    a = (1 - conf) / 2

    def cdf(nc: float) -> float:
        v = float(nct.cdf(t, df, nc))
        if np.isnan(v):  # far tail of the bracket: P(T < t | ncp) is ~0 above t, ~1 below
            return 0.0 if nc > t else 1.0
        return v

    def solve(q: float) -> float:
        return optimize.brentq(lambda nc: cdf(nc) - q, t - 8, t + 8)

    return t / np.sqrt(n), solve(1 - a) / np.sqrt(n), solve(a) / np.sqrt(n)


def group_bmc(log_evidence: np.ndarray) -> dict[str, np.ndarray]:
    """Random-effects group Bayesian model selection (``groupBMC``) of a models x participants log-evidence matrix
    (or a proxy such as ``-0.5 * AICc``): per model, the mean and variance of the expected frequency and the
    exceedance and protected exceedance probabilities."""
    from groupBMC.groupBMC import GroupBMC

    res = GroupBMC(log_evidence).get_result()
    out = {"frequency_mean": res.frequency_mean, "frequency_var": res.frequency_var,
           "exceedance_probability": res.exceedance_probability,
           "protected_exceedance_probability": res.protected_exceedance_probability}
    return {k: np.asarray(v, dtype=float).reshape(-1) for k, v in out.items()}


def compute_consistency(
    df: pd.DataFrame,
    *,
    stim_col: str,
    rep_col: str,
    participant_col: str = "participant",
    incentive_col: str = "incentive",
    choice_col: str = "choice",
) -> pd.DataFrame:
    """``df`` with a ``consistency`` column: on each repeated presentation (``rep_col == 1``), 1 when the choice equals
    the choice at the first original presentation (``rep_col == 0``, first in ``df`` order) of the same participant,
    stimulus and incentive and 0 otherwise; NaN on the original presentations and on repeats without an original.
    Choices may be coded {-1, +1} or {0, 1}."""
    df = df.copy()
    keys = [participant_col, stim_col, incentive_col]
    is_orig = df[rep_col] == 0
    is_repeat = df[rep_col] == 1

    ref = (
        df.loc[is_orig, keys + [choice_col]]
        .drop_duplicates(subset=keys, keep="first")
        .rename(columns={choice_col: "_ref_choice"})
    )
    # merge the reference onto the repeats without disturbing the row order of df
    repeats = df.loc[is_repeat, keys + [choice_col]].copy()
    repeats = repeats.merge(ref, on=keys, how="left")

    def _to01(x):
        x = x.astype(float)
        ux = np.unique(x[~np.isnan(x)])
        if set(ux).issubset({-1.0, 1.0}):
            return (x + 1.0) / 2.0
        return x

    r_choice = _to01(repeats[choice_col].to_numpy())
    r_ref = _to01(repeats["_ref_choice"].to_numpy())
    cons = (r_choice == r_ref).astype("float")
    cons[~np.isfinite(r_ref)] = np.nan  # repeat without an original presentation

    df["consistency"] = np.nan
    df.loc[is_repeat, "consistency"] = cons
    return df


def fit_slopes_intercepts(
    df,
    dv: str = "choice_var",
    iv: str = "seqlen",
    group_vars: list[str] | None = None,
    iv_center: float | None = None,
):
    """Fit ``DV ~ (IV - c)`` per group (default: participant × incentive).

    When ``iv_center`` is omitted and ``iv == 'seqlen'``, ``c`` is
    :data:`CHOICE_VAR_SEQLEN_CENTER_FOR_SLOPE` (8). The **intercept** is the
    fitted level at ``IV == c``; **slope** is the change in DV per unit raw IV
    (same as an uncentered ``DV ~ IV`` fit). Pass ``iv_center=0`` for a raw
    intercept at ``IV == 0``.
    """
    if group_vars is None:
        group_vars = ["participant", "incentive"]
    if iv_center is None and iv == "seqlen":
        center: float | None = float(CHOICE_VAR_SEQLEN_CENTER_FOR_SLOPE)
    elif iv_center is not None:
        center = float(iv_center)
    else:
        center = None

    results = []

    for keys, sub in df.groupby(group_vars):
        if not isinstance(keys, tuple):
            keys = (keys,)
        if sub[iv].nunique() < 2:
            continue

        sub_clean = sub[[iv, dv]].dropna()
        if sub_clean.empty:
            continue

        y = sub_clean[dv]
        x = sub_clean[iv].astype(float)
        if center is not None:
            x = x - center
        X = sm.add_constant(x.to_numpy())
        model = sm.OLS(y, X).fit()
        slope = float(model.params.iloc[1])
        intercept = float(model.params.iloc[0])

        row: dict = {
            group_vars[0]: keys[0],
            "slope": slope,
            "intercept": intercept,
            "r2": model.rsquared,
            "n": len(sub_clean),
        }
        if len(group_vars) > 1:
            # Legacy: second group key always stored as ``incentive``.
            row["incentive"] = keys[1]
        results.append(row)

    return pd.DataFrame(results)
