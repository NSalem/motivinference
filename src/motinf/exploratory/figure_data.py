"""Data preparation for the paper's figures (``scripts/exploratory/figures``).

One section per figure, so that each figure's numbers come from the package rather than from another script. No
P value is computed here: the stars of the figures are read from the tables they belong to (:func:`paper_holm_p`,
or the ``p_holm`` / ``p_corr`` columns written by the stats and tables stages).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from motinf.exploratory.variance_partition import log_decomposition_partition

LOSS, NEUTRAL, GAIN = -1.0, 0.0, 1.0  # incentive codes, as in the data
INCENTIVES = (LOSS, NEUTRAL, GAIN)
INCENTIVE_CODES = [-1, 0, 1]  # the same as integers (columns of the per-participant pivots, ``incentive`` of summaries)
PAPER_TABLES = Path("results/exploratory/merged/paper_tables/paper_tables.json")


@lru_cache(maxsize=None)
def _paper_tables(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def paper_holm_p(root: Path, measure: str, sample: str = "merged") -> dict[float, float]:
    """Holm-corrected P of loss − neutral and gain − neutral for a measure of Tables 2 and 3 and Supplementary
    Table 7, as the tables print it (``paper_tables.json`` → ``table2`` → sample → measure)."""
    m = _paper_tables(Path(root) / PAPER_TABLES)["table2"][sample][measure]
    return {LOSS: m["loss_neutral"]["p_holm"], GAIN: m["gain_neutral"]["p_holm"]}


def paper_bias_holm_p(root: Path, measure: str, sample: str = "merged") -> dict[float, float]:
    """The same for a common bias of Supplementary Table 6 (``supp`` → ``biases`` → sample → measure: ``cardinal``,
    ``serial``, ``b_0``, ``b_prev_stim``, ``b_prev_resp``, ``b_prev_key``)."""
    m = _paper_tables(Path(root) / PAPER_TABLES)["supp"]["biases"][sample][measure]
    return {LOSS: m["loss_neutral"]["p_holm"], GAIN: m["gain_neutral"]["p_holm"]}


# ----------------------------------------------------------------------------------------------------------- #
# Figure 2d: choice variability over sequence length, data and inference-only model
# ----------------------------------------------------------------------------------------------------------- #


def _sem_nan(x: np.ndarray) -> float:
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    n = int(v.size)
    if n < 2:
        return 0.0
    return float(np.nanstd(v, ddof=1) / np.sqrt(n))


def _group_choice_var_summary(
    df: pd.DataFrame,
    group_cols: list[str],
    beh_col: str,
    pred_col: str,
    *,
    beh: str,
    pred: str,
) -> pd.DataFrame:
    """Per group, mean ± s.e.m. of the data (``beh_col``) and of the model (``pred_col``) over participants, as
    ``<beh>_mean``, ``<beh>_sem``, ``<beh>_low`` / ``<beh>_high`` (mean ∓ s.e.m.) and the same for ``pred``."""
    rows: list[dict[str, Any]] = []
    for key, g in df.groupby(group_cols, sort=False):
        if isinstance(key, tuple):
            row: dict[str, Any] = {c: key[i] for i, c in enumerate(group_cols)}
        else:
            row = {group_cols[0]: key}
        for name, col in ((beh, beh_col), (pred, pred_col)):
            v = g[col].to_numpy(dtype=float, copy=False)
            mean, sem = float(np.nanmean(v)), _sem_nan(v)
            row.update({f"{name}_mean": mean, f"{name}_sem": sem, f"{name}_low": mean - sem,
                        f"{name}_high": mean + sem})
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(group_cols).reset_index(drop=True)


def summary_choice_var_pooled_from_merged_cleaned(m: pd.DataFrame) -> pd.DataFrame:
    """Pooled incentive × seqlen natural-scale summary (mean ± s.e.m.) of the data and the model, from the frame of
    ``choice_var_seqlen_preprocess.merge_and_preprocess_choice_var_beh_pred``."""
    need = {"participant", "incentive", "seqlen", "choice_var", "pred_choice_var"}
    if m.empty or not need.issubset(m.columns):
        return pd.DataFrame()
    return _group_choice_var_summary(m, ["incentive", "seqlen"], "choice_var", "pred_choice_var",
                                     beh="beh_natural", pred="pred_natural")


def choice_var_incentive_effect_long_by_seqlen(m: pd.DataFrame) -> pd.DataFrame:
    """Per participant × seqlen × incentive(±1): neutral-vs-incentive effect, beh & model.

    Pairs each incentivized row (gain=+1, loss=-1) with that same participant's
    neutral (incentive=0) row at the same seqlen, from the frame of
    ``choice_var_seqlen_preprocess.merge_and_preprocess_choice_var_beh_pred`` (participants with an extreme cell
    already excluded). Returns the natural-scale reduction (neutral − incentive; positive = variance shrinks under
    incentive) for behaviour (``beh_red``) and model (``pred_red``).
    """
    beh_col, pred_col = "choice_var", "pred_choice_var"
    need = {"participant", "incentive", "seqlen", beh_col, pred_col}
    if m.empty or not need.issubset(m.columns):
        return pd.DataFrame()
    d = m[["participant", "incentive", "seqlen", beh_col, pred_col]].copy()
    d["participant"] = d["participant"].astype(str)
    d["incentive"] = d["incentive"].astype(float)
    d["seqlen"] = d["seqlen"].astype(int)

    neutral = d[d["incentive"] == 0.0].rename(columns={beh_col: "beh0", pred_col: "pred0"})
    neutral = neutral[["participant", "seqlen", "beh0", "pred0"]]

    chunks: list[pd.DataFrame] = []
    for inc in (-1.0, 1.0):
        incentivized = d[d["incentive"] == inc].rename(columns={beh_col: "beh1", pred_col: "pred1"})
        incentivized = incentivized[["participant", "seqlen", "beh1", "pred1"]]
        merged = neutral.merge(incentivized, on=["participant", "seqlen"], how="inner")
        if merged.empty:
            continue
        merged["incentive"] = inc
        merged["beh_red"] = merged["beh0"] - merged["beh1"]
        merged["pred_red"] = merged["pred0"] - merged["pred1"]
        chunks.append(merged)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def summary_choice_var_incentive_effect(long_df: pd.DataFrame) -> pd.DataFrame:
    """Group (incentive, seqlen) summary (mean ± s.e.m.) of the per-participant natural-scale reduction
    (``beh_red`` / ``pred_red``, :func:`choice_var_incentive_effect_long_by_seqlen`): columns ``beh_red_mean``,
    ``beh_red_sem``, ``beh_red_low``, ``beh_red_high`` and the same for ``pred_red`` (Figure 2d)."""
    need = {"incentive", "seqlen", "beh_red", "pred_red"}
    if long_df.empty or not need.issubset(long_df.columns):
        return pd.DataFrame()
    return _group_choice_var_summary(long_df, ["incentive", "seqlen"], "beh_red", "pred_red",
                                     beh="beh_red", pred="pred_red")


# ----------------------------------------------------------------------------------------------------------- #
# Figure 3 and Supplementary Figure 6: full model
# ----------------------------------------------------------------------------------------------------------- #


CHANNELS = ("ang", "inf", "sel")


def motiv_effect_per_channel_long(df: pd.DataFrame) -> pd.DataFrame:
    """``(participant, seqlen, ang, inf, sel)`` = ``var_red[full] - var_red[ko_ch]`` per channel.

    ``df`` must have columns ``participant``, ``seqlen``, ``model_id``, ``var_red`` and cover
    ``model_id`` in ``{full, ko_ang, ko_inf, ko_sel}``. Differenced at the per-participant level
    (not on already-aggregated means) so a later mean/SEM reflects the paired covariance between
    the "full" and "ko_ch" estimates for the same participant. Exact given the model's additive
    variance decomposition (``ko_ch`` only zeroes that channel's ``motiv_ch``): summing the three
    resulting channel columns reproduces the "full" ``var_red`` exactly.
    """
    d = df.copy()
    d["participant"] = d["participant"].astype(str)
    ko_ids = {f"ko_{ch}" for ch in CHANNELS}
    needed = {"full", *ko_ids}
    have = set(d["model_id"].astype(str).unique())
    if not needed.issubset(have):
        raise ValueError(f"missing model_id(s) {needed - have}")
    wide = d[d["model_id"].astype(str).isin(needed)].pivot_table(
        index=["participant", "seqlen"], columns="model_id", values="var_red"
    )
    return pd.DataFrame({ch: wide["full"] - wide[f"ko_{ch}"] for ch in CHANNELS}).reset_index()


def integer_percents_largest_remainder(p: np.ndarray) -> list[int]:
    """Map nonnegative proportions to integer percentages that sum to exactly 100.

    Avoids independent rounding (e.g. 34+34+33) that can sum to 101.
    """
    v = np.asarray(p, dtype=float)
    v = np.where(np.isfinite(v) & (v > 0), v, 0.0)
    s = float(v.sum())
    if s <= 0:
        return [0] * int(v.size)
    x = (v / s) * 100.0
    base = np.floor(x).astype(int)
    rem = x - base.astype(float)
    need = int(100 - int(base.sum()))
    if need == 0:
        return base.tolist()
    if need > 0:
        order = np.argsort(-rem, kind="stable")
        for k in range(need):
            base[order[k % order.size]] += 1
        return base.tolist()
    # need < 0: trim from smallest remainders (rare float noise)
    order = np.argsort(rem, kind="stable")
    k = 0
    while need < 0 and k < base.size:
        if base[order[k % order.size]] > 0:
            base[order[k % order.size]] -= 1
            need += 1
        k += 1
    return base.tolist()


def variance_partition_log_decomposition(
    df: pd.DataFrame,
    channels: tuple[str, ...],
    *,
    n_boot: int = 4000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Panel 1's per-channel bars, log decomposition, with participant-bootstrap SEMs.

    From ``total_var`` in ``variance_partition_params.csv``: at |incentive| = 0 each channel's
    bar is its mean share of the participant's baseline total; the |incentive| = 1 bar's length
    is the geometric-mean total ratio (incentive / baseline), split by each channel's average
    contribution to the participants' change in log total variance — per participant, that
    channel's share of the total times its own log-variance change ``-2 * motiv_X`` (see
    :func:`motinf.exploratory.variance_partition.log_decomposition_partition`). A channel
    without incentive modulation keeps its baseline segment. Averaging per-participant
    ``total_var_prop_baseline`` instead is biased upward: a channel's ratio is
    ``exp(-2 * motiv_X)``, which a few participants with negative ``motiv_X`` push far above 1.

    SEMs are the std of the bootstrap replicates (participants resampled with both
    |incentive| levels together, as for panel 2's knockout SEM), for the baseline bar, the
    incentive bar and their difference. Returns ``(p0, p1, sem0, sem1, d_sem, boot)`` with
    ``boot`` of shape (n_boot, n_channels, 2).
    """
    wide = df.pivot_table(index="participant", columns=["channel", "incentive_abs"], values="total_var")
    comp = np.stack(
        [wide[[(ch, 0), (ch, 1)]].to_numpy(dtype=float) for ch in channels], axis=1
    )  # participant x channel x |incentive|
    bar, boot, _ = log_decomposition_partition(comp, ref=0, n_boot=n_boot, seed=seed)
    return (
        bar[:, 0],
        bar[:, 1],
        boot[:, :, 0].std(axis=0),
        boot[:, :, 1].std(axis=0),
        (boot[:, :, 0] - boot[:, :, 1]).std(axis=0),
        boot,
    )


def knockout_channel_share_point_and_sem(
    df_pred: pd.DataFrame,
    channels: tuple[str, ...],
    *,
    n_boot: int = 4000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Point estimate + SEM for each channel's share of the incentive effect, from
    paired ``ko_X - full`` contrasts on natural-scale predicted choice variance
    (``knockout_predictions.csv``'s ``pred_choice_var``), restricted to
    ``|incentive|=1`` trials (knocking out ``motiv_X`` only changes predictions at
    nonzero incentive, so there's no need to subtract a baseline level here). Both
    ``full`` and ``ko_X`` share the same participant's baseline parameters, so this
    isolates channel X's own ``motiv_X`` contribution directly — no per-participant
    division by a noisy baseline-composition denominator.

    Natural scale, mean across participants (raw units, plain mean; the natural scale is
    skewed, so a few participants with large channel-specific effects pull the mean). SEM is
    the std of a participant-bootstrap distribution of the same share statistic: the share is
    a channel's effect divided by a sum that includes it, so a closed-form SEM treating the
    sum as fixed would be wrong. In Experiment 2 alone this SEM can exceed the point estimate.
    """
    nan3 = np.full(len(channels), np.nan)
    need_cols = {"full", *(f"ko_{ch}" for ch in channels)}
    d = df_pred.copy()
    d["incentive_abs"] = d["incentive"].astype(float).abs().round().astype(int)
    sub = (
        d[d["incentive_abs"] == 1]
        .groupby(["participant", "model_id"])["pred_choice_var"]
        .mean()
        .reset_index()
    )
    wide = sub.pivot(index="participant", columns="model_id", values="pred_choice_var")
    if not need_cols.issubset(wide.columns):
        return nan3, nan3

    deltas = pd.DataFrame({ch: wide[f"ko_{ch}"] - wide["full"] for ch in channels})
    vals = deltas.to_numpy(dtype=float)
    n = vals.shape[0]

    mean_d = np.maximum(0.0, np.nanmean(vals, axis=0))
    s = float(mean_d.sum())
    point = mean_d / s if s > 1e-12 else nan3.copy()
    if n < 3:
        return point, nan3

    rng = np.random.default_rng(seed)
    boot = np.full((n_boot, len(channels)), np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        md = np.maximum(0.0, np.nanmean(vals[idx], axis=0))
        sb = float(md.sum())
        if sb > 1e-12:
            boot[b] = md / sb
    sem = np.nanstd(boot, axis=0)
    return point, sem


# ----------------------------------------------------------------------------------------------------------- #
# Figure 4: noise-bias partition
# ----------------------------------------------------------------------------------------------------------- #


_COMPONENTS = ("noise_var", "bias_var")


def tag_participant_col(df: pd.DataFrame, exp: str) -> pd.DataFrame:
    """Make participant IDs unique across concatenated experiments."""
    out = df.copy()
    out["participant"] = out["participant"].astype(str) + f"__{exp}"
    return out


def _complete_var_wide(df: pd.DataFrame, cols: list[str], inc_order: list[int]) -> dict[str, np.ndarray]:
    """Participant x incentive arrays (columns in ``inc_order``) for each of ``cols``.

    Only participants with a finite value in every column at every incentive are kept, so
    each bootstrap resample draws whole participants.
    """
    d = df[["participant", "incentive", *cols]].copy()
    d["incentive"] = pd.to_numeric(d["incentive"], errors="coerce").round().astype("Int64")
    wide = d.pivot(index="participant", columns="incentive", values=cols)
    wide = wide.loc[:, [(c, inc) for c in cols for inc in inc_order]].dropna()
    return {c: wide[c].to_numpy(dtype=float) for c in cols}


def _noise_bias_partition(
    df: pd.DataFrame, *, n_boot: int = 5000, seed: int = 0
) -> tuple[np.ndarray, np.ndarray | None, int]:
    """Log decomposition of noise / bias variance over incentives −1, 0, +1, relative to £0.

    ``bar[k, j]`` is component k (``_COMPONENTS``: noise, bias) at incentive
    ``INCENTIVE_CODES[j]`` as a fraction of the £0 total. The £0 bar is the mean
    per-participant composition; each incentive bar's length is the geometric-mean change in
    total variance vs £0, split by each component's average contribution to the participants'
    change in log total variance (see
    :func:`motinf.exploratory.variance_partition.log_decomposition_partition`). ``boot``
    holds participant-bootstrap replicates of ``bar`` (None if ``n_boot == 0``); ``n`` is the
    number of participants used.
    """
    arr = _complete_var_wide(df, list(_COMPONENTS), INCENTIVE_CODES)
    comp = np.stack([arr[c] for c in _COMPONENTS], axis=1)
    bar, boot, keep = log_decomposition_partition(
        comp, INCENTIVE_CODES.index(0), n_boot=n_boot, seed=seed
    )
    return bar, boot, int(keep.sum())


def wide_by_incentive(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Participants x incentive (float columns -1, 0, 1) of ``col``, complete cases only."""
    d = df[["participant", "incentive", col]].copy()
    d["incentive"] = pd.to_numeric(d["incentive"], errors="coerce").astype(float)
    return d.pivot(index="participant", columns="incentive", values=col).dropna()


def prop_of_baseline_summary(df: pd.DataFrame, *, n_boot: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Stacked-bar segments per incentive (``_noise_bias_partition``) with 95% bootstrap CIs."""
    bar, boot, n = _noise_bias_partition(df, n_boot=n_boot, seed=seed)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    summ = pd.DataFrame({"incentive": INCENTIVE_CODES, "n": n})
    for k, name in enumerate(("noise", "bias")):
        summ[f"{name}_m"], summ[f"{name}_lo"], summ[f"{name}_hi"] = bar[k], lo[k], hi[k]
    return summ


# ----------------------------------------------------------------------------------------------------------- #
# Figure 5: response times
# ----------------------------------------------------------------------------------------------------------- #


def holm_p(tests: pd.DataFrame, key: str, value) -> float:
    row = tests[tests[key] == value]
    return float(row["p_holm"].iloc[0]) if len(row) and "p_holm" in row else np.nan


def cell_p(tests: pd.DataFrame, family: str, coef: str, incentive: str, seqlen: float) -> float:
    row = tests[(tests["family"] == family) & (tests["coef"] == coef) & (tests["incentive"] == incentive) & np.isclose(tests["seqlen"], seqlen)]
    return float(row["p_holm"].iloc[0]) if len(row) else np.nan


def cell_diff_units(coef: str, ref_mean: float, d: np.ndarray) -> tuple[float, float]:
    """Incentive − neutral difference for display: intercept as the ms gap between geometric-mean RTs at zero
    evidence (SEM by the delta method), slope in native units (log RT per LLR)."""
    md, sed = float(np.mean(d)), float(np.std(d, ddof=1) / np.sqrt(len(d)))
    if coef == "intercept":
        return 1000 * (np.exp(ref_mean + md) - np.exp(ref_mean)), 1000 * np.exp(ref_mean + md) * sed
    return md, sed


def cell_linear_wide(coefs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {c: coefs.pivot_table(index="participant", columns=["incentive", "seqlen"], values=c) for c in ("intercept", "slope")}


def cell_linear_curves(coefs: pd.DataFrame) -> dict[tuple[float, float], tuple[np.ndarray, ...]]:
    """(seqlen, incentive) -> (evidence grid, geometric-mean fitted RT, lower, upper) with ± between-participant
    SEM of the fitted log RT, over each seqlen's 2.5–97.5% evidence range."""
    W = cell_linear_wide(coefs)
    n = len(W["intercept"])
    ranges = coefs.groupby("seqlen")[["ev_lo", "ev_hi"]].first()
    curves = {}
    for s in ranges.index:
        grid = np.linspace(ranges.loc[s, "ev_lo"], ranges.loc[s, "ev_hi"], 150)
        for inc in INCENTIVES:
            P = W["intercept"][(inc, s)].to_numpy(float)[:, None] + W["slope"][(inc, s)].to_numpy(float)[:, None] * grid[None, :]
            m, se = P.mean(0), P.std(0, ddof=1) / np.sqrt(n)
            curves[(s, inc)] = (grid, np.exp(m), np.exp(m - se), np.exp(m + se))
    return curves


def cell_linear_points(bins: pd.DataFrame) -> pd.DataFrame:
    """Binned data points: mean log RT (``m``) ± between-participant SEM (``se``) per seqlen × incentive × bin, at
    the bin's mean evidence (``x``); only bins where ≥ 50% of participants have the cell in all three incentives."""
    n_part = bins["participant"].nunique()
    points = []
    for (s, b), cb in bins.groupby(["seqlen", "ev_bin"]):
        w = cb.pivot_table(index="participant", columns="incentive", values="log_rt").dropna()
        if len(w) < 0.5 * n_part or w.shape[1] < 3:
            continue
        xs = cb.groupby("incentive")["ev_mean"].mean()
        for inc in INCENTIVES:
            points.append({"seqlen": s, "incentive": inc, "x": xs[inc], "m": w[inc].mean(), "se": w[inc].std(ddof=1) / np.sqrt(len(w))})
    return pd.DataFrame(points)


# ----------------------------------------------------------------------------------------------------------- #
# Supplementary Figure 2b: recovery of the inference-only model
# ----------------------------------------------------------------------------------------------------------- #


def load_recovery_runs(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path)
    need = {"ok", "sd_inf_true", "sd_inf_fit", "incentive", "participant"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"recovery_runs.csv missing columns: {sorted(missing)}")
    return df


def default_recovery_runs_csv(root: Path, exp_name: str) -> Path:
    return Path(root) / "results" / "exploratory" / exp_name / "modeling" / "recovery_noise_per_cond" / "recovery_runs.csv"


def participant_incentive_pivots(df_ok: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean σ_inf across replicates per participant × incentive (true and fit)."""
    pivot_true = (
        df_ok.groupby(["participant", "incentive"], as_index=False)
        .agg(sd_inf_true_mean=("sd_inf_true", "mean"))
        .pivot(index="participant", columns="incentive", values="sd_inf_true_mean")
        .reindex(columns=INCENTIVE_CODES)
    )
    pivot_fit = (
        df_ok.groupby(["participant", "incentive"], as_index=False)
        .agg(sd_inf_fit_mean=("sd_inf_fit", "mean"))
        .pivot(index="participant", columns="incentive", values="sd_inf_fit_mean")
        .reindex(columns=INCENTIVE_CODES)
    )
    mask = pivot_true.notna().all(axis=1) & pivot_fit.notna().all(axis=1)
    return pivot_true[mask], pivot_fit[mask]


def logdiff_effects_long(pivot_true: pd.DataFrame, pivot_fit: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-12
    gen_rows: list[dict] = []
    for inc in (-1, 1):
        g0 = np.log(np.clip(pivot_true[0].to_numpy(dtype=float), eps, None))
        gi = np.log(np.clip(pivot_true[inc].to_numpy(dtype=float), eps, None))
        f0 = np.log(np.clip(pivot_fit[0].to_numpy(dtype=float), eps, None))
        fi = np.log(np.clip(pivot_fit[inc].to_numpy(dtype=float), eps, None))
        for pid, x, y in zip(pivot_true.index.astype(str), gi - g0, fi - f0):
            gen_rows.append({"participant": pid, "incentive": inc, "gen": x, "fit": y})
    return pd.DataFrame(gen_rows)


# ----------------------------------------------------------------------------------------------------------- #
# Supplementary Figure 5: full-model recovery
# ----------------------------------------------------------------------------------------------------------- #


PARNAMES = ["sd_ang", "sd_inf", "sd_sel", "motiv_ang", "motiv_inf", "motiv_sel"]


RECOVERED_COLUMN = {"sd_ang": "fixed_sd_ang_from_estim", "motiv_ang": "fixed_motiv_ang_from_estim"}


def bootstrap_corr_sem(
    x: np.ndarray, y: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """Pearson correlation and its bootstrap SEM (std of the bootstrap r's)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    n = int(len(x))
    if n < 2:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)

    r_hat = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(r_hat):
        return float("nan"), float("nan")

    idx = rng.integers(0, n, size=(n_boot, n))
    xb = x[idx]
    yb = y[idx]

    xb0 = xb - xb.mean(axis=1, keepdims=True)
    yb0 = yb - yb.mean(axis=1, keepdims=True)
    num = np.sum(xb0 * yb0, axis=1)
    den = np.sqrt(np.sum(xb0**2, axis=1) * np.sum(yb0**2, axis=1))
    r_boot = num / np.clip(den, 1e-20, np.inf)

    sem = float(np.std(r_boot, ddof=1)) if n_boot > 1 else float("nan")
    return r_hat, sem
