"""
Bias / noise variance partition.

:func:`compute_variance_partition` splits estimation errors with the test-retest correlation over
repeated stimuli; :func:`fit_bias_share_repeats` is the repeated-sequence likelihood fit of the
Integration-task bias share; :func:`log_decomposition_partition` is the group summary used for every
stacked variance-partition bar (behavioural noise / bias and the full model's channels).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.polynomial.hermite_e import hermegauss
from pingouin import circ_corrcc
from scipy import optimize, special, stats


def wrap_semicircle(x: np.ndarray) -> np.ndarray:
    """
    Wrap angles on a semicircle [-pi/2, pi/2] given radians.
    """
    x = np.mod(x, np.pi)
    x[x > np.pi / 2] -= np.pi
    return x


def compute_variance_partition(
    df: pd.DataFrame,
    participant_col: str = "participant",
    incentive_col: str = "incentive",
    stim_col: str = "stim_id_unique",
    resp_rad_col: str = "resp_estim",
    true_ang_col: str = "estimang_noisy",
) -> pd.DataFrame:
    """
    Compute bias / noise variance partition for estimation errors (cleaned ``data_clean_est.csv`` trials).

    Parameters
    ----------
    df
        Estimation trials (one row per trial).
    participant_col, incentive_col, stim_col
        Column names for participant, incentive and stimulus identity.
    resp_rad_col
        Column with responses in radians.
    true_ang_col
        Column with true angles in radians.

    Returns
    -------
    pd.DataFrame
        One row per participant × incentive with:
        - n_trials, n_pairs, absdiff
        - rho_testretest (raw circular test-retest correlation), rho (bias share: the
          correlation clipped to ``RHO_BOUNDS``), total_var, bias_var, noise_var
        - bias_sd, noise_sd, total_sd
    """
    dfe = df.copy()

    # Convert units & compute wrapped errors on the semicircle
    # Both response and true angles are expected in radians.
    dfe["resp_estim_rad"] = dfe[resp_rad_col].to_numpy(float)
    dfe["true_ang_rad"] = dfe[true_ang_col].to_numpy(float)
    dfe["resp_estim_wrapped"] = np.mod(dfe["resp_estim_rad"], np.pi)
    dfe["true_ang_wrapped"] = np.mod(dfe["true_ang_rad"], np.pi)
    dfe["error_wrapped"] = wrap_semicircle(
        dfe["resp_estim_wrapped"].to_numpy(float)
        - dfe["true_ang_wrapped"].to_numpy(float)
    )

    # Stable order key (to form pairs)
    dfe = dfe.reset_index(names="orig_index")

    group_cols = [participant_col, incentive_col]
    summ_rows = []

    def make_pairs(group: pd.DataFrame) -> pd.DataFrame | None:
        g = group.sort_values("orig_index")
        n = len(g)
        if n < 2:
            return None
        idx1 = list(range(0, n - 1, 2))
        idx2 = [i + 1 for i in idx1]
        return pd.DataFrame(
            {
                "e1": g.iloc[idx1]["error_wrapped"].to_numpy(),
                "e2": g.iloc[idx2]["error_wrapped"].to_numpy(),
            }
        )

    for keys, sub in dfe.groupby(group_cols, observed=True):
        errs = pd.to_numeric(sub["error_wrapped"], errors="coerce")
        total_var = np.var(errs.to_numpy()) if errs.shape[0] > 0 else np.nan

        pairs_list = []
        for _stim_id, g in sub.groupby(stim_col, observed=True):
            p = make_pairs(g)
            if p is not None:
                pairs_list.append(p)

        if len(pairs_list) == 0:
            rho = np.nan
            n_pairs = 0
            absdiff_mean = np.nan
        else:
            pairs = pd.concat(pairs_list, ignore_index=True)
            x = 2 * pairs["e1"] + np.pi
            y = 2 * pairs["e2"] + np.pi
            rho = circ_corrcc(x, y)[0] if pairs.shape[0] >= 2 else np.nan
            absdiff = np.abs(pairs["e1"] - pairs["e2"])
            n_pairs = int(pairs.shape[0])
            absdiff_mean = float(absdiff.mean())

        split = split_total_variance(rho, total_var)
        summ_rows.append(
            {
                participant_col: keys[0],
                incentive_col: keys[1],
                "n_trials": int(sub.shape[0]),
                "n_pairs": n_pairs,
                "absdiff": absdiff_mean,
                "rho_testretest": rho,
                "rho": float(split["rho"]),
                "total_var": total_var,
                "bias_var": float(split["bias_var"]),
                "noise_var": float(split["noise_var"]),
                "bias_sd": float(split["bias_sd"]),
                "noise_sd": float(split["noise_sd"]),
                "total_sd": float(split["total_sd"]),
            }
        )

    return pd.DataFrame(summ_rows)


# Bounds on the bias share and floor on the component SDs used when saving a partition, so no
# stored component is exactly 0 (logs stay defined) and no bias share is exactly 1
RHO_BOUNDS = (1e-3, 0.999)
SD_FLOOR = 1e-4


def split_total_variance(
    rho,
    total_var,
    *,
    rho_bounds: tuple[float, float] = RHO_BOUNDS,
    sd_floor: float = SD_FLOOR,
) -> dict:
    """Split total variance into bias and noise given the bias share ``rho``.

    ``rho`` (the correlation between repeated presentations = bias share) is clipped to
    ``rho_bounds``; ``bias_var = rho * total_var`` and ``noise_var = (1 - rho) * total_var``;
    the bias, noise and total SDs are floored at ``sd_floor`` and the variances recomputed from
    the floored SDs. NaNs propagate. Works elementwise on scalars or arrays.
    """
    rho_c = np.clip(np.asarray(rho, dtype=float), *rho_bounds)
    tv = np.asarray(total_var, dtype=float)
    out = {"rho": rho_c}
    for name, var in (("bias", rho_c * tv), ("noise", (1.0 - rho_c) * tv), ("total", tv)):
        sd = np.fmax(np.sqrt(np.clip(var, 0.0, None)), sd_floor)
        sd = np.where(np.isnan(var), np.nan, sd)
        out[f"{name}_sd"] = sd
        if name != "total":
            out[f"{name}_var"] = sd**2
    return out


# Grid of bias shares on which each cell's likelihood is evaluated (and stored)
BIAS_SHARE_GRID = np.linspace(0.0, 0.995, 200)
_GH_X, _GH_W = hermegauss(120)
_GH_W = _GH_W / np.sqrt(2 * np.pi)  # weights for E[f(Z)], Z ~ N(0, 1)


def _repeat_loglik(
    rho: np.ndarray | float,
    pairs: tuple[np.ndarray, np.ndarray],
    multi: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Log-likelihood of one cell's repeated presentations at each value of ``rho``.

    ``pairs = (h, n_a)``: sequences shown twice, with standardized evidence h and the number of
    'A' choices (0, 1 or 2). ``multi = (h, n_a, n)``: sequences shown n >= 3 times.
    """
    rho = np.atleast_1d(np.asarray(rho, dtype=float))
    ll = np.zeros(rho.shape)
    h, n_a = pairs
    if h.size:
        # P(A, A) = Phi(h) - 2 T(h, a), P(B, B) = Phi(-h) - 2 T(h, a), P(A, B) = P(B, A) = 2 T(h, a),
        # with Owen's T and a = sqrt((1 - rho) / (1 + rho))
        t2 = 2 * special.owens_t(h[None, :], np.sqrt((1 - rho) / (1 + rho))[:, None])
        pa = stats.norm.cdf(h)[None, :]
        p = np.where(n_a == 2, pa - t2, np.where(n_a == 0, 1 - pa - t2, t2))
        ll += np.log(np.clip(p, 1e-300, None)).sum(axis=1)
    h, n_a, n = multi
    if h.size:
        # conditional on the shared (bias) draw the presentations are independent
        z = (h[None, :, None] + np.sqrt(rho)[:, None, None] * _GH_X) / np.sqrt(1 - rho)[:, None, None]
        logp = n_a[None, :, None] * stats.norm.logcdf(z) + (n - n_a)[None, :, None] * stats.norm.logcdf(-z)
        top = logp.max(axis=2, keepdims=True)
        ll += (top[..., 0] + np.log(np.exp(logp - top) @ _GH_W)).sum(axis=1)
    return ll


def fit_bias_share_repeats(
    h: np.ndarray,
    choose_a: np.ndarray,
    group: np.ndarray,
    *,
    prior: tuple[float, float] = (1.1, 1.1),
    grid: np.ndarray = BIAS_SHARE_GRID,
) -> dict:
    """
    Bias share ``rho`` of the inference noise, fitted from choices to repeated sequences.

    Model (Drugowitsch et al., 2016, Neuron, Supplement Eq. S23): on each presentation the
    decision variable is ``h + b + v``, in units of the fitted inference noise SD. The bias
    ``b`` is identical on every presentation of a sequence, the noise ``v`` is fresh, and
    ``rho = var(b) / (var(b) + var(v))`` is both the bias share of the variance and the
    correlation between presentations (bias variance = rho x sigma_inf^2, not rho^2).

    ``h``: standardized evidence of each trial (model mean / fitted noise SD, the same on every
    presentation of a sequence); ``choose_a``: whether the trial chose the option favoured by
    positive evidence; ``group``: sequence identity. Sequences shown once carry no information
    about rho and are ignored. Pairs use the exact bivariate-normal probability; sequences shown
    3+ times a 120-node Gauss-Hermite integral over the shared draw (log error < 4e-4 for
    rho <= 0.95). The likelihood is evaluated on ``grid`` and refined with bounded Brent.

    Returns a dict: ``rho_map`` (posterior mode under a Beta(*prior) prior, which keeps it off
    0), ``rho_mle``, ``ci_lo`` / ``ci_hi`` (profile-likelihood 95% interval of the likelihood
    alone), ``n_pairs`` / ``n_multi`` (sequences shown twice / 3+ times) and ``loglik`` on
    ``grid`` (for later shrinkage or diagnostics).
    """
    grid = np.asarray(grid, dtype=float)
    df = pd.DataFrame({"h": np.asarray(h, float), "a": np.asarray(choose_a, bool), "g": np.asarray(group)})
    agg = df.groupby("g", sort=False).agg(
        h=("h", "first"), h_min=("h", "min"), h_max=("h", "max"), n=("a", "size"), n_a=("a", "sum")
    )
    if np.any(agg["h_max"] - agg["h_min"] > 1e-8 * np.maximum(1.0, agg["h"].abs())):
        raise ValueError("Standardized evidence differs between presentations of the same sequence.")
    two, many = agg[agg["n"] == 2], agg[agg["n"] >= 3]
    pairs = (two["h"].to_numpy(float), two["n_a"].to_numpy(int))
    multi = (many["h"].to_numpy(float), many["n_a"].to_numpy(float), many["n"].to_numpy(float))
    out = {"n_pairs": int(len(two)), "n_multi": int(len(many))}
    if len(two) + len(many) == 0:
        return {**out, "rho_map": np.nan, "rho_mle": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "loglik": np.full(grid.shape, np.nan)}

    ll = _repeat_loglik(grid, pairs, multi)
    a_prior, b_prior = prior

    def ll_at(r: float) -> float:
        return float(_repeat_loglik(r, pairs, multi)[0])

    def log_prior(r):
        r = np.asarray(r, dtype=float)
        with np.errstate(divide="ignore"):
            return (a_prior - 1) * np.log(r) + (b_prior - 1) * np.log1p(-r)

    def refine(values: np.ndarray, f, lower: float) -> float:
        """Best grid point, refined by bounded Brent within its neighbouring grid cells."""
        i = int(np.argmax(values))
        lo, hi = max(grid[max(i - 1, 0)], lower), grid[min(i + 1, len(grid) - 1)]
        if hi <= lo:
            return float(grid[i])
        res = optimize.minimize_scalar(lambda r: -f(r), bounds=(lo, hi), method="bounded", options={"xatol": 1e-7})
        return float(res.x) if -res.fun >= values[i] else float(grid[i])

    rho_mle = refine(ll, ll_at, lower=0.0)
    rho_map = refine(ll + log_prior(grid), lambda r: ll_at(r) + float(log_prior(r)), lower=1e-9)

    # profile-likelihood 95% interval: rho with ll >= max - chi2(0.95, 1) / 2 (linear interpolation)
    thr = max(ll.max(), ll_at(rho_mle)) - stats.chi2.ppf(0.95, 1) / 2
    idx = np.flatnonzero(ll >= thr)

    def crossing(i_in: int, i_out: int) -> float:
        x0, x1, y0, y1 = grid[i_in], grid[i_out], ll[i_in], ll[i_out]
        return float(x0 + (thr - y0) * (x1 - x0) / (y1 - y0))

    ci_lo = float(grid[0]) if idx[0] == 0 else crossing(idx[0], idx[0] - 1)
    ci_hi = float(grid[-1]) if idx[-1] == len(grid) - 1 else crossing(idx[-1], idx[-1] + 1)
    return {**out, "rho_map": rho_map, "rho_mle": rho_mle, "ci_lo": ci_lo, "ci_hi": ci_hi, "loglik": ll}


def _expm1_over_x(x: np.ndarray) -> np.ndarray:
    """``(exp(x) - 1) / x``, equal to 1 at ``x == 0``: L(e^x, 1) for the logarithmic mean L."""
    x = np.asarray(x, dtype=float)
    safe = np.where(x == 0, 1.0, x)
    return np.where(x == 0, 1.0, np.expm1(safe) / safe)


def log_decomposition_partition(
    comp_var: np.ndarray,
    ref: int,
    *,
    n_boot: int = 5000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """
    Group summary of a variance partition across conditions, for stacked bars relative to the
    reference condition: log decomposition.

    ``comp_var`` has shape (n_participants, n_components, n_conditions): each participant's
    variance per component (noise / bias, or the ang / inf / sel channels) and condition, the
    components adding up to that participant's total T in that condition. ``ref`` indexes the
    reference (neutral) condition.

    Per participant i, the change in log total variance is split exactly into components:

        ln(T_i[c] / T_i[ref]) = sum_k d_i[k, c],   d_i[k, c] = (comp_i[k, c] - comp_i[k, ref]) / L_i[c]

    with L_i[c] the logarithmic mean of T_i[c] and T_i[ref] (L(a, b) = (a - b) / ln(a / b),
    L(a, a) = a). Equivalently, d_i[k, c] is component k's own change in log variance weighted
    by its (log-mean) share of the total, so an unchanged component contributes exactly 0, and
    only totals are logged (components may be 0). Averaging over participants, each counting
    equally:

        share[k]  = mean_i comp_i[k, ref] / T_i[ref]          reference bar: mean composition
        G[c]      = exp(mean_i ln(T_i[c] / T_i[ref]))          bar length: geometric-mean ratio
        bar[k, c] = share[k] + L(G[c], 1) * mean_i d_i[k, c]

    Bars are fractions of the reference total. The reference bar sums to 1; the bar at c sums to
    G[c] (sum_k mean_i d = ln G and L(G, 1) ln G = G - 1), the point estimate of a paired test on
    log total variance; with one participant, bar[k, c] is exactly comp[k, c] / T[ref]. Averaging
    raw ratios comp_i[k, c] / T_i[ref] instead is biased upward (a participant whose reference
    total comes out low gets a huge ratio), ratios of group means are dominated by the few
    participants with the largest variances, and scaling the mean composition at c by G moves
    components that did not change. This is the logarithmic-mean weighting of LMDI (log-mean
    Divisia index) decompositions.

    Participants with a non-finite component or a non-positive total in any condition are dropped.

    Returns ``(bar, boot, keep)``: ``bar`` (n_components, n_conditions); ``boot``
    (n_boot, n_components, n_conditions), participant-bootstrap replicates of ``bar`` (whole
    participants resampled, all conditions together), or None when ``n_boot == 0``; ``keep``, the
    boolean mask of participants used.
    """
    comp = np.asarray(comp_var, dtype=float)
    if comp.ndim != 3:
        raise ValueError(f"comp_var must be 3-D (participant x component x condition), got {comp.shape}")
    total = comp.sum(axis=1)
    keep = np.all(np.isfinite(comp), axis=(1, 2)) & np.all(total > 0, axis=1)
    comp, total = comp[keep], total[keep]
    n = total.shape[0]
    if n == 0:
        raise ValueError("No participant has finite components and a positive total in every condition.")
    t_ref = total[:, [ref]]
    log_ratio = np.log(total / t_ref)  # participant x condition
    log_mean = t_ref * _expm1_over_x(log_ratio)  # L(T_c, T_ref), computed stably
    contrib = (comp - comp[:, :, [ref]]) / log_mean[:, None, :]  # participant x component x condition
    share = comp[:, :, ref] / t_ref  # participant x component

    def _bar(w: np.ndarray) -> np.ndarray:
        # w: participant weights summing to 1, shape (n,) or (n_boot, n)
        to_bar_units = _expm1_over_x(w @ log_ratio)  # L(G, 1)
        return np.einsum("...i,ik->...k", w, share)[..., :, None] + to_bar_units[
            ..., None, :
        ] * np.einsum("...i,ikc->...kc", w, contrib)

    bar = _bar(np.full(n, 1.0 / n))
    boot = None
    if n_boot > 0:
        # multinomial counts / n == weights of a resample-with-replacement of participants
        w = np.random.default_rng(seed).multinomial(n, np.full(n, 1.0 / n), size=n_boot) / n
        boot = _bar(w)
    return bar, boot, keep

