"""Log RT as a function of choice-aligned evidence, incentive and sequence length (Integration task).

Evidence toward the choice: ``ev_chosen = sumllr_noisy x choice`` (positive = the displayed evidence
supported the choice made, negative = it opposed it).

- Per-cell linear fits (Figure 5B, Supplementary Table 8): OLS ``log(RT) ~ 1 + ev_chosen`` separately per
  participant x incentive x sequence length (intercept = log RT at zero evidence, slope = log RT per LLR
  unit). Evidence and sequence length interact (the RT-evidence slope is much steeper for short
  sequences), so each length gets its own intercept and slope. Tests are one-sample t-tests on
  per-participant values or differences (dz with exact CI); families: slope vs 0 (Holm over all
  incentives x lengths), incentive - neutral per coefficient (Holm over gain/loss x lengths), and
  length pairs within each coefficient x incentive level or incentive difference (Holm over the 3 pairs).
  Binned mean log RT per participant x incentive x length (cells with < ``MIN_TRIALS_PER_CELL`` trials
  missing) gives the data points drawn under the fits.
- Binned model (incentive x evidence interaction quoted with Supplementary Table 8): evidence binned at
  ``EVIDENCE_BIN_EDGES`` (right-closed); per participant one OLS ``log(RT) ~ 0 + incentive x bin cell
  indicators + seqlen indicators`` (sequence-length effect shared across cells, reference length
  ``SEQLEN_REF``), cells with < ``MIN_TRIALS_PER_CELL`` trials dropped with their trials; incentive x bin
  rmANOVA on participants with every cell. Each effect's Greenhouse-Geisser epsilon is computed from its
  own orthonormal contrasts (pingouin's two-way interaction epsilon is inaccurate when both factors have
  more than two levels).

Writes ``rt_evidence_cell_linear_{coefs,bins,tests}.csv`` and ``rt_evidence_anova.csv`` to the RT
analysis stats directory (see ``rt_analysis.py``).
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import helmert

from motinf.exploratory.choice_var_seqlen_preprocess import CHOICE_VAR_SEQLEN_CENTER_FOR_SLOPE
from motinf.exploratory.stats import dz_ci, holm

EVIDENCE_BIN_EDGES = (-np.inf, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, np.inf)
MIN_TRIALS_PER_CELL = 5
SEQLEN_REF = float(CHOICE_VAR_SEQLEN_CENTER_FOR_SLOPE)
INCENTIVE_LEVELS = (-1.0, 0.0, 1.0)


def evidence_bin_labels(edges: tuple[float, ...] = EVIDENCE_BIN_EDGES) -> list[str]:
    """Comma-free labels for right-closed bins: ``≤ -1``, ``-1 to 0``, …, ``> 4``."""
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if not np.isfinite(lo):
            labels.append(f"≤ {hi:g}")
        elif not np.isfinite(hi):
            labels.append(f"> {lo:g}")
        else:
            labels.append(f"{lo:g} to {hi:g}")
    return labels


def _prepare_trials(df_inf: pd.DataFrame, edges: tuple[float, ...]) -> pd.DataFrame:
    df = df_inf.copy()
    for col in ["rt", "choice", "sumllr_noisy", "seqlen", "incentive"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["rt"] = df["rt"].replace([np.inf, -np.inf], np.nan)
    df = df.loc[df["rt"].notna() & (df["rt"] > 0) & df["incentive"].isin(INCENTIVE_LEVELS)].copy()
    df = df.dropna(subset=["participant", "choice", "sumllr_noisy", "seqlen"])
    df["log_rt"] = np.log(df["rt"].to_numpy(float))
    df["ev_chosen"] = df["sumllr_noisy"].to_numpy(float) * df["choice"].to_numpy(float)
    df["ev_bin"] = np.digitize(df["ev_chosen"].to_numpy(float), np.asarray(edges[1:-1]), right=True)
    return df


def _fit_participant_cells(d: pd.DataFrame, n_bins: int, min_trials: int) -> list[dict]:
    """One OLS per participant: a free mean per incentive × bin cell plus shared seqlen offsets
    (indicator per non-reference seqlen), so cell coefficients are log RT at ``SEQLEN_REF``.
    Returns one row per cell."""
    inc_idx = d["incentive"].map({inc: i for i, inc in enumerate(INCENTIVE_LEVELS)}).to_numpy(int)
    cell = inc_idx * n_bins + d["ev_bin"].to_numpy(int)
    n_cells = len(INCENTIVE_LEVELS) * n_bins
    counts = np.bincount(cell, minlength=n_cells)
    kept_cells = np.flatnonzero(counts >= min_trials)
    rows_kept = np.isin(cell, kept_cells)

    cell_k = cell[rows_kept]
    seqlen_k = d["seqlen"].to_numpy(float)[rows_kept]
    X_cells = (cell_k[:, None] == kept_cells[None, :]).astype(float)
    seq_levels = [s for s in np.unique(seqlen_k) if s != SEQLEN_REF]
    X_seq = np.column_stack([(seqlen_k == s).astype(float) for s in seq_levels]) if seq_levels else np.empty((len(cell_k), 0))
    beta, *_ = np.linalg.lstsq(np.column_stack([X_cells, X_seq]), d["log_rt"].to_numpy(float)[rows_kept], rcond=None)

    log_rt = np.full(n_cells, np.nan)
    log_rt[kept_cells] = beta[: len(kept_cells)]
    ev = d["ev_chosen"].to_numpy(float)
    ev_mean = np.full(n_cells, np.nan)
    for c in kept_cells:
        ev_mean[c] = float(ev[cell == c].mean())

    rows: list[dict] = []
    for i, inc in enumerate(INCENTIVE_LEVELS):
        for b in range(n_bins):
            c = i * n_bins + b
            rows.append(
                {
                    "incentive": inc,
                    "ev_bin": b,
                    "n_trials": int(counts[c]),
                    "ev_mean": ev_mean[c],
                    "log_rt": log_rt[c],
                }
            )
    return rows


def _fit_cells(df_trials: pd.DataFrame, edges: tuple[float, ...], min_trials: int) -> pd.DataFrame:
    n_bins = len(edges) - 1
    labels = evidence_bin_labels(edges)
    rows: list[dict] = []
    for pid, d in df_trials.groupby("participant", observed=True):
        for r in _fit_participant_cells(d, n_bins, min_trials):
            rows.append({"participant": pid, **r, "ev_bin_label": labels[r["ev_bin"]]})
    return pd.DataFrame(rows)


def _two_way_rm_anova(cells: pd.DataFrame, factors: tuple[str, str], dv: str = "log_rt") -> pd.DataFrame:
    """Two-way within-participant rmANOVA on participants with every cell, via each effect's
    orthonormal contrasts: same SS/F as the univariate ANOVA, plus the correct per-effect
    Greenhouse-Geisser epsilon (pingouin's interaction epsilon is off for > 2 × > 2 designs)."""
    fa, fb = factors
    wide = cells.pivot_table(index="participant", columns=[fa, fb], values=dv).dropna()
    n = len(wide)
    if n < 3:
        return pd.DataFrame()
    levels_a = sorted(wide.columns.get_level_values(0).unique())
    levels_b = sorted(wide.columns.get_level_values(1).unique())
    a, b = len(levels_a), len(levels_b)
    Y = np.column_stack([wide[(i, j)].to_numpy(float) for i in levels_a for j in levels_b])
    ca, cb = helmert(a, full=False).T, helmert(b, full=False).T
    mean_a, mean_b = np.full((a, 1), 1 / np.sqrt(a)), np.full((b, 1), 1 / np.sqrt(b))
    effects = {
        fa: np.kron(ca, mean_b),
        fb: np.kron(mean_a, cb),
        f"{fa} * {fb}": np.kron(ca, cb),
    }
    rows: list[dict] = []
    for source, C in effects.items():
        Z = Y @ C
        k = C.shape[1]
        z_mean = Z.mean(axis=0)
        ss = float(n * np.sum(z_mean**2))
        ss_err = float(np.sum((Z - z_mean) ** 2))
        df1, df2 = k, k * (n - 1)
        F = (ss / df1) / (ss_err / df2)
        S = np.atleast_2d(np.cov(Z, rowvar=False))
        eps = float(np.trace(S) ** 2 / (k * np.trace(S @ S)))
        rows.append(
            {
                "Source": source,
                "SS": ss,
                "ddof1": df1,
                "ddof2": df2,
                "MS": ss / df1,
                "F": F,
                "p-unc": float(stats.f.sf(F, df1, df2)),
                "p-GG-corr": float(stats.f.sf(F, eps * df1, eps * df2)),
                "np2": ss / (ss + ss_err),
                "eps": eps,
                "n_participants": n,
            }
        )
    return pd.DataFrame(rows)


def _paired_stats(diff: np.ndarray) -> dict[str, float]:
    """One-sample t on per-participant log-RT differences: mean with 95% CI, % RT change,
    t, p and dz with exact CI."""
    d = np.asarray(diff, float)
    d = d[np.isfinite(d)]
    n = len(d)
    m, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(n))
    t = m / se
    t_crit = float(stats.t.ppf(0.975, n - 1))
    dz, dz_lo, dz_hi = dz_ci(t, n)
    return {
        "n": n,
        "mean_diff": m,
        "ci_lo": m - t_crit * se,
        "ci_hi": m + t_crit * se,
        "pct_rt": 100.0 * (np.exp(m) - 1.0),
        "t": t,
        "df": n - 1,
        "p": float(2.0 * stats.t.sf(abs(t), n - 1)),
        "dz": dz,
        "dz_lo": dz_lo,
        "dz_hi": dz_hi,
    }


def _fit_cell_linear(df_trials: pd.DataFrame) -> pd.DataFrame:
    """OLS ``log RT ~ 1 + ev_chosen`` per participant × incentive × seqlen. ``ev_lo``/``ev_hi``: the
    2.5/97.5% quantiles of ev_chosen at that seqlen (pooled), the range the fitted curves are drawn over."""
    rows: list[dict] = []
    for (pid, inc, s), g in df_trials.groupby(["participant", "incentive", "seqlen"], observed=True):
        a = b = np.nan
        if len(g) >= 3:
            X = np.column_stack([np.ones(len(g)), g["ev_chosen"].to_numpy(float)])
            a, b = np.linalg.lstsq(X, g["log_rt"].to_numpy(float), rcond=None)[0]
        rows.append({"participant": pid, "incentive": float(inc), "seqlen": float(s), "n_trials": len(g), "intercept": a, "slope": b})
    coefs = pd.DataFrame(rows)
    q = df_trials.groupby("seqlen")["ev_chosen"].quantile([0.025, 0.975]).unstack()
    coefs["ev_lo"] = coefs["seqlen"].map(q[0.025])
    coefs["ev_hi"] = coefs["seqlen"].map(q[0.975])
    return coefs


def _cell_linear_bins(df_trials: pd.DataFrame, min_trials: int) -> pd.DataFrame:
    """Mean log RT per participant × incentive × seqlen × evidence bin (NaN below ``min_trials``): the
    data points drawn under the per-cell linear fits."""
    b = (
        df_trials.groupby(["participant", "incentive", "seqlen", "ev_bin"], observed=True)
        .agg(n_trials=("log_rt", "size"), ev_mean=("ev_chosen", "mean"), log_rt=("log_rt", "mean"))
        .reset_index()
    )
    b.loc[b["n_trials"] < min_trials, "log_rt"] = np.nan
    return b


def _cell_linear_tests(coefs: pd.DataFrame) -> pd.DataFrame:
    """Tests on the per-cell linear coefficients (log units; one row per test, ``holm_family`` = the set
    Holm-corrected together): slope vs 0 (all incentives × seqlens); incentive − neutral per coefficient
    (gain/loss × seqlens); seqlen pairs within each coefficient × incentive level or incentive difference."""
    names = {-1.0: "loss", 0.0: "neutral", 1.0: "gain"}
    W = {c: coefs.pivot_table(index="participant", columns=["incentive", "seqlen"], values=c) for c in ("intercept", "slope")}
    seqlens = sorted(coefs["seqlen"].unique())
    rows: list[dict] = []

    def add(family: str, holm_family: str, coef: str, incentive: str, seqlen: float, comparison: str, d) -> None:
        rows.append(
            {"family": family, "holm_family": holm_family, "coef": coef, "incentive": incentive, "seqlen": seqlen,
             "comparison": comparison, **_paired_stats(np.asarray(d, float))}
        )

    for inc in INCENTIVE_LEVELS:
        for s in seqlens:
            add("slope_vs_zero", "slope_vs_zero", "slope", names[inc], s, "vs zero", W["slope"][(inc, s)])
    for c in ("intercept", "slope"):
        for inc in (1.0, -1.0):
            for s in seqlens:
                add("incentive_vs_neutral", f"incentive_vs_neutral|{c}", c, f"{names[inc]}_minus_neutral", s, "vs neutral",
                    W[c][(inc, s)] - W[c][(0.0, s)])
        series = {names[inc]: {s: W[c][(inc, s)] for s in seqlens} for inc in INCENTIVE_LEVELS}
        for inc in (1.0, -1.0):
            series[f"{names[inc]}_minus_neutral"] = {s: W[c][(inc, s)] - W[c][(0.0, s)] for s in seqlens}
        for label, by_s in series.items():
            for s1, s2 in combinations(seqlens, 2):
                add("seqlen_pairs", f"seqlen_pairs|{c}|{label}", c, label, np.nan, f"{s1:g} vs {s2:g}", by_s[s1] - by_s[s2])
    out = pd.DataFrame(rows)
    out["p_holm"] = out.groupby("holm_family")["p"].transform(lambda p: pd.Series(holm(p.tolist()), index=p.index))
    return out


def run_rt_evidence_regression(df_inf: pd.DataFrame, stats_dir: Path) -> None:
    """Fit the per-cell linear and binned RT-evidence models on trial data and write their tables."""
    stats_dir.mkdir(parents=True, exist_ok=True)
    df_trials = _prepare_trials(df_inf, EVIDENCE_BIN_EDGES)
    cells = _fit_cells(df_trials, EVIDENCE_BIN_EDGES, MIN_TRIALS_PER_CELL)
    cell_coefs = _fit_cell_linear(df_trials)
    tables = {
        "rt_evidence_anova": _two_way_rm_anova(cells, ("incentive", "ev_bin")),
        "rt_evidence_cell_linear_coefs": cell_coefs,
        "rt_evidence_cell_linear_bins": _cell_linear_bins(df_trials, MIN_TRIALS_PER_CELL),
        "rt_evidence_cell_linear_tests": _cell_linear_tests(cell_coefs),
    }
    for name, table in tables.items():
        table.to_csv(stats_dir / f"{name}.csv", index=False)
