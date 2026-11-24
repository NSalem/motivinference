"""Robust (Huber) linear regression on z-scored predictors, with the weighted R² of the fit."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import zscore as scipy_zscore


def r2_weighted(y: np.ndarray, yhat: np.ndarray, w: np.ndarray) -> float:
    """R² with the residual and total sums of squares weighted by ``w`` (the final IRLS weights of a robust fit)."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    w = np.asarray(w, dtype=float)
    if w.sum() <= 0:
        return np.nan
    ybar = float(np.sum(w * y) / np.sum(w))
    ss_res = float(np.sum(w * (y - yhat) ** 2))
    ss_tot = float(np.sum(w * (y - ybar) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def fit_rlm_zscored_predictors(
    y: pd.Series,
    X: pd.DataFrame,
    *,
    zscore_cols: list[str] | None = None,
) -> tuple[Any, np.ndarray]:
    """Fit Huber RLM with optional z-scored predictors and intercept; outcome on natural scale."""
    x_arr = X.to_numpy(float)
    if zscore_cols is None:
        x_proc = scipy_zscore(x_arr, nan_policy="omit")
    else:
        x_proc = x_arr.copy()
        zset = set(zscore_cols)
        for j, col in enumerate(X.columns):
            if col in zset:
                x_proc[:, j] = scipy_zscore(x_arr[:, j], nan_policy="omit")
    Xc = sm.add_constant(x_proc)
    res = sm.RLM(endog=y.to_numpy(float), exog=Xc).fit()
    return res, Xc


def fit_rlm_block(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    *,
    zscore_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Huber RLM of ``y_col`` on ``x_cols`` (complete cases; every predictor z-scored, or only ``zscore_cols``) with an
    intercept. Returns ``n``, ``r2_full`` (:func:`r2_weighted`) and ``rows``: coefficient, SE, z and p of the intercept
    (``predictor`` = ``const``) and of each predictor, in that order."""
    df_local = df[[y_col, *x_cols]].dropna()
    n = int(df_local.shape[0])
    min_n = len(x_cols) + 2
    if n < min_n:
        return {"n": n, "r2_full": np.nan, "rows": []}

    y = df_local[y_col]
    try:
        res, Xc = fit_rlm_zscored_predictors(y, df_local[x_cols], zscore_cols=zscore_cols)
    except (ValueError, np.linalg.LinAlgError):
        return {"n": n, "r2_full": np.nan, "rows": []}

    r2_full = r2_weighted(y.to_numpy(float), res.predict(Xc), res.weights)

    params = np.asarray(res.params, dtype=float).ravel()
    bse = np.asarray(res.bse, dtype=float).ravel()
    try:
        pvals = np.asarray(res.pvalues, dtype=float).ravel()
    except (AttributeError, TypeError, ValueError):
        pvals = np.full(params.shape, np.nan, dtype=float)
    if pvals.shape != params.shape:
        pvals = np.full_like(params, np.nan, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        zvals = np.where(np.isfinite(bse) & (bse > 0), params / bse, np.nan)

    rows: list[dict[str, Any]] = []
    for i, predictor in enumerate(["const", *x_cols]):
        rows.append(
            {
                "predictor": predictor,
                "coef": float(params[i]) if i < len(params) else np.nan,
                "se": float(bse[i]) if i < len(bse) else np.nan,
                "z": float(zvals[i]) if i < len(zvals) else np.nan,
                "pvalue": float(pvals[i]) if i < len(pvals) else np.nan,
            }
        )

    return {"n": n, "r2_full": float(r2_full), "rows": rows}
