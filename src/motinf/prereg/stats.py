"""Preregistered copy of required statistics helpers."""

import numpy as np
import pandas as pd
from statsmodels import api as sm


def probit_sigma2(x, y):
    y = np.asarray(y).astype(float)
    if set(np.unique(y)).issubset({-1.0, 1.0}):
        y = (y + 1.0) / 2.0

    X = np.asarray(x).reshape(-1, 1)
    X = sm.add_constant(X)
    model = sm.Probit(y, X)
    res = model.fit(disp=False)
    beta1 = res.params[1]
    sigma2 = (1.0 / beta1) ** 2
    return sigma2, res


def fit_slopes_intercepts(df, dv="choice_var", iv="seqlen", group_vars=None, verbose=False):
    if group_vars is None:
        group_vars = ["participant", "incentive"]

    results = []
    n_groups_total = 0
    skipped_too_few_levels = 0
    skipped_missing_values = 0
    skipped_fit_error = 0

    for (pid, inc), sub in df.groupby(group_vars):
        n_groups_total += 1
        sub_clean = sub[[iv, dv]].dropna()

        if sub_clean.empty:
            skipped_missing_values += 1
            if verbose:
                print(f"[fit_slopes_intercepts] skip participant={pid}, incentive={inc}: all rows missing {iv}/{dv}.")
            continue

        if sub[iv].nunique() < 2:
            skipped_too_few_levels += 1
            if verbose:
                print(
                    f"[fit_slopes_intercepts] skip participant={pid}, incentive={inc}: "
                    f"{iv} has <2 unique levels (n_unique={sub[iv].nunique()})."
                )
            continue

        try:
            X = sm.add_constant(sub_clean[iv])
            y = sub_clean[dv]
            model = sm.OLS(y, X).fit()
            slope = model.params.get(iv, np.nan)
            intercept = model.params.get("const", np.nan)
            if not np.isfinite(slope) or not np.isfinite(intercept):
                skipped_fit_error += 1
                if verbose:
                    print(
                        f"[fit_slopes_intercepts] skip participant={pid}, incentive={inc}: "
                        "non-finite slope/intercept."
                    )
                continue

            results.append(
                {
                    "participant": pid,
                    "incentive": inc,
                    "slope": slope,
                    "intercept": intercept,
                    "r2": model.rsquared,
                    "n": len(sub_clean),
                }
            )
        except Exception as exc:
            skipped_fit_error += 1
            if verbose:
                print(
                    f"[fit_slopes_intercepts] skip participant={pid}, incentive={inc}: "
                    f"OLS failed with {type(exc).__name__}: {exc}"
                )

    if verbose:
        n_kept = len(results)
        print(
            "[fit_slopes_intercepts] summary: "
            f"total_groups={n_groups_total}, kept={n_kept}, "
            f"skipped_too_few_levels={skipped_too_few_levels}, "
            f"skipped_missing_values={skipped_missing_values}, "
            f"skipped_fit_error={skipped_fit_error}"
        )

    return pd.DataFrame(results)


def sliding_psychometric(
    x,
    y,
    x_min=-6.0,
    x_max=6.0,
    step=0.1,
    window=0.5,
    kernel="uniform",
    bandwidth=None,
    counts_min=5,
    return_ci=True,
    ci=0.95,
):
    """Non-parametric psychometric curve via sliding window."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    uy = np.unique(y[np.isfinite(y)])
    if set(uy).issubset({-1.0, 1.0}):
        y = (y + 1.0) / 2.0
    elif not set(uy).issubset({0.0, 1.0}):
        raise ValueError("y must be in {-1,+1} or {0,1}")

    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]

    grid = np.arange(x_min, x_max + 1e-12, step)
    mgrid = grid.size

    halfw = window / 2.0
    if kernel not in ("uniform", "gaussian"):
        raise ValueError("kernel must be 'uniform' or 'gaussian'")
    if kernel == "gaussian" and bandwidth is None:
        bandwidth = max(1e-12, window / 2.0)

    p = np.full(mgrid, np.nan)
    n = np.zeros(mgrid)
    n_eff = np.zeros(mgrid)

    if return_ci:
        from scipy.stats import norm

        z = norm.ppf(0.5 + ci / 2.0)
        lo = np.full(mgrid, np.nan)
        hi = np.full(mgrid, np.nan)
    else:
        lo = hi = None

    for i, c in enumerate(grid):
        if kernel == "uniform":
            mask = (x >= c - halfw) & (x < c + halfw)
            w = mask.astype(float)
        else:
            u = (x - c) / bandwidth
            w = np.exp(-0.5 * u * u)

        sw = w.sum()
        if sw <= 0:
            continue

        n_eff_i = (sw**2) / np.sum(w**2) if np.any(w > 0) else 0.0
        if n_eff_i < counts_min:
            n[i] = sw
            n_eff[i] = n_eff_i
            continue

        p_hat = (w * y).sum() / sw
        p[i] = p_hat
        n[i] = sw
        n_eff[i] = n_eff_i

        if return_ci:
            ne = n_eff_i
            denom = 1.0 + (z**2) / ne
            center = p_hat + (z**2) / (2 * ne)
            margin = z * np.sqrt((p_hat * (1 - p_hat) / ne) + (z**2) / (4 * ne**2))
            lo[i] = max(0.0, (center - margin) / denom)
            hi[i] = min(1.0, (center + margin) / denom)

    return {"grid": grid, "p": p, "n": n, "n_eff": n_eff, "lo": lo, "hi": hi}
