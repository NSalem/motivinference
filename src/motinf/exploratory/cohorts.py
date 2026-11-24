"""Helpers for pooling the two cohorts (exp1 + exp2) into one analysis.

``--exp-name merged`` stands for both experiments. Participant identifiers are then prefixed as
``{segment}__{original_id}`` so the same raw id in two experiments does not collide.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MERGED = "merged"


def cohort_subexperiments(exp_name: str) -> list[str]:
    """``["exp1", "exp2"]`` for ``merged`` (any case), otherwise the one experiment named."""
    key = exp_name.strip()
    return ["exp1", "exp2"] if key.lower() == MERGED else [key]


def participant_prefixed(segment: str, pid: str) -> str:
    return f"{segment}__{str(pid)}"


def _prefix_participant(df: pd.DataFrame, segment: str) -> pd.DataFrame:
    out = df.copy()
    out["participant"] = out["participant"].astype(str).map(lambda p, s=segment: participant_prefixed(s, p))
    return out


def _merge_prereg_behavior_table(root: Path, segments: list[str], name: str) -> pd.DataFrame:
    """Concatenate ``results/prereg/<seg>/behavior/<name>`` over the segments.

    When ``len(segments) > 1``, participant ids are prefixed ``seg__<id>`` to avoid collisions.
    A single segment leaves participant ids unchanged.
    """
    prefix_pids = len(segments) > 1
    chunks: list[pd.DataFrame] = []
    for seg in segments:
        path = root / "results" / "prereg" / seg / "behavior" / name
        if not path.exists():
            raise FileNotFoundError(f"Missing behaviour table for merge: {path}")
        df = pd.read_csv(path)
        if "participant" not in df.columns:
            raise KeyError(f"{path} missing participant column")
        chunks.append(_prefix_participant(df, seg) if prefix_pids else df.copy())
    return pd.concat(chunks, ignore_index=True)


def merge_behavior_mean_inc(root: Path, segments: list[str]) -> pd.DataFrame:
    """Per participant and incentive: the pre-registered behaviour summary ``data_mean_inc.csv``."""
    return _merge_prereg_behavior_table(root, segments, "data_mean_inc.csv")


def merge_behavior_seqlen(root: Path, segments: list[str]) -> pd.DataFrame:
    """Per participant, incentive and sequence length: ``data_mean_inc_seqlen.csv``."""
    return _merge_prereg_behavior_table(root, segments, "data_mean_inc_seqlen.csv")


def apply_row_limits(df: pd.DataFrame, max_ntrials: int, max_nsess: int) -> pd.DataFrame:
    """Trials of the first ``max_nsess`` sessions and with ``trialN <= max_ntrials`` (0: no limit), for quick runs."""
    out = df
    if max_nsess > 0:
        out = out[pd.to_numeric(out["session"], errors="coerce") <= max_nsess]
    if max_ntrials > 0:
        out = out[pd.to_numeric(out["trialN"], errors="coerce") <= max_ntrials]
    return out.reset_index(drop=True)


_SEQ_COLS: tuple[str, ...] = tuple(f"seqang_noisy_{i}" for i in range(1, 13))


def _merge_processed(
    root: Path,
    segments: list[str],
    task: str,
    required: tuple[str, ...],
    *,
    keep: list[str] | None = None,
) -> pd.DataFrame:
    """Concatenate ``data/processed/<seg>/data_clean_<task>.csv`` over the segments.

    Every file must have the ``required`` columns. ``keep``: the columns to return, in that order; otherwise all
    columns, which must then be the same set in every segment (the first segment fixes their order). Participant ids
    become ``str``, prefixed ``seg__<id>`` when there are several segments; ``incentive`` becomes float and, when
    present, ``seqlen`` int.
    """
    prefix_pids = len(segments) > 1
    chunks: list[pd.DataFrame] = []
    ref_cols: list[str] | None = None
    for seg in segments:
        path = root / "data" / "processed" / seg / f"data_clean_{task}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing trial data: {path}")
        df = pd.read_csv(path)
        missing = sorted(set(required).difference(df.columns))
        if missing:
            raise KeyError(f"{path} missing required columns: {missing}")
        if keep is not None:
            df = df[keep]
        elif ref_cols is None:
            ref_cols = list(df.columns)
        elif set(df.columns) != set(ref_cols):
            raise ValueError(f"{path}: columns differ from those of {segments[0]!r}.")
        else:
            df = df[ref_cols]
        out = _prefix_participant(df, seg) if prefix_pids else df.copy()
        if not prefix_pids:
            out["participant"] = out["participant"].astype(str)
        out["incentive"] = out["incentive"].astype(float)
        if "seqlen" in out.columns:
            out["seqlen"] = out["seqlen"].astype(int)
        chunks.append(out)
    return pd.concat(chunks, ignore_index=True)


def merge_inf_trials_for_choice_var(root: Path, segments: list[str]) -> pd.DataFrame:
    """Sequence Categorization trials (``data_clean_inf.csv``), the columns the choice-variability predictions need."""
    keep = ["participant", "incentive", "seqlen", "sumllr_noisy", "choice", *_SEQ_COLS]
    return _merge_processed(root, segments, "inf", tuple(keep), keep=keep)


def merge_data_clean_inf(root: Path, segments: list[str]) -> pd.DataFrame:
    """Sequence Categorization trials (``data_clean_inf.csv``), all columns (those of the model's trial predictions
    required: ``df_to_pack`` / ``get_ppost_fast``)."""
    return _merge_processed(root, segments, "inf", ("participant", "seqlen", "incentive", "choice", *_SEQ_COLS))


def merge_data_clean_est(root: Path, segments: list[str]) -> pd.DataFrame:
    """Stimulus Reproduction trials (``data_clean_est.csv``), all columns."""
    return _merge_processed(root, segments, "est",
                            ("participant", "incentive", "estimang_noisy", "dev", "resp_estim", "dev_noisy"))


def merge_sd_inf_long(root: Path, segments: list[str]) -> pd.DataFrame:
    """Concatenate ``results/prereg/<seg>/model_fit/stats/sd_inf_long.csv``: the inference noise of the
    pre-registered inference-only model per participant and incentive, as ``scripts/prereg/run_model_stats.py``
    extracts it from ``fit_noise_per_cond.npz``.

    Columns ``participant``, ``incentive``, ``sd_inf``, ``sd_inf_log``. Floats are parsed round-trip, so the values
    are bit-identical to those in the npz. When ``len(segments) > 1``, participant ids are prefixed ``seg__<id>``.
    """
    prefix_pids = len(segments) > 1
    chunks: list[pd.DataFrame] = []
    for seg in segments:
        path = root / "results" / "prereg" / seg / "model_fit" / "stats" / "sd_inf_long.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing inference noise per condition (run scripts/prereg/run_model_stats.py): {path}")
        df = pd.read_csv(path, dtype={"participant": str}, float_precision="round_trip")
        chunks.append(_prefix_participant(df, seg) if prefix_pids else df)
    return pd.concat(chunks, ignore_index=True)


def validate_prereg_inputs_exist(root: Path, segments: list[str]) -> None:
    """Raise if any required prereg / processed file for a segment is missing."""
    for seg in segments:
        for rel in (
            ("results", "prereg", seg, "model_fit", "stats", "sd_inf_long.csv"),
            ("results", "prereg", seg, "behavior", "data_mean_inc.csv"),
            ("results", "prereg", seg, "behavior", "data_mean_inc_seqlen.csv"),
            ("data", "processed", seg, "data_clean_inf.csv"),
        ):
            p = root.joinpath(*rel)
            if not p.exists():
                raise FileNotFoundError(f"Missing required input for cohort segment {seg!r}: {p}")


def validate_full_model_inputs_exist(root: Path, segments: list[str]) -> None:
    """Raise if prereg/processed prerequisites or per-segment ``fit_full_model.npz`` is missing."""
    validate_prereg_inputs_exist(root, segments)
    for seg in segments:
        npz_path = (
            root / "results" / "exploratory" / seg / "modeling" / "fit_files" / "fit_full_model.npz"
        )
        if not npz_path.exists():
            raise FileNotFoundError(
                f"Missing full-model NPZ for cohort segment {seg!r}: {npz_path}"
            )


def sd_inf_rt_split_long_from_npz(
    npz_path: Path,
    *,
    segment: str | None = None,
) -> pd.DataFrame:
    """Expand one RT-split NPZ into long participant × rt_bin × incentive ``sd_inf`` rows."""
    res = np.load(npz_path, allow_pickle=True)
    participants = np.asarray(res["participants"], dtype=object)
    rt_bins_raw = np.asarray(res["rt_bins"], dtype=object)
    pars = res["pars_inf_rt_split"]

    rows: list[dict[str, float | str | int]] = []
    for ip, participant in enumerate(participants):
        pid_str = participant_prefixed(segment, str(participant)) if segment else str(participant)
        for jb, rt_bin in enumerate(rt_bins_raw):
            theta = pars[ip, jb]
            if theta is None:
                continue
            theta_arr = np.asarray(theta, dtype=float).ravel()
            if theta_arr.size != 3 or np.any(~np.isfinite(theta_arr)):
                continue
            sd_inf_n, sd_inf_g, sd_inf_l = map(float, theta_arr)
            rows.extend(
                [
                    {"participant": pid_str, "rt_bin": str(rt_bin), "incentive": -1, "sd_inf": sd_inf_l},
                    {"participant": pid_str, "rt_bin": str(rt_bin), "incentive": 0, "sd_inf": sd_inf_n},
                    {"participant": pid_str, "rt_bin": str(rt_bin), "incentive": 1, "sd_inf": sd_inf_g},
                ]
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["participant", "rt_bin", "incentive", "sd_inf", "sd_inf_log"])
    df["sd_inf_log"] = np.log(df["sd_inf"])
    return df


def merge_sd_inf_rt_split_long(root: Path, segments: list[str]) -> pd.DataFrame:
    """Long RT-split ``sd_inf`` table from each segment's ``fit_inf_rt_split.npz``."""
    prefix_pids = len(segments) > 1
    chunks: list[pd.DataFrame] = []
    for seg in segments:
        npz_path = root / "results" / "exploratory" / seg / "modeling" / "fit_files" / "fit_inf_rt_split.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing RT-split inference fit (run fit_inf_per_cond_rt_split.py): {npz_path}")
        chunks.append(sd_inf_rt_split_long_from_npz(npz_path, segment=seg if prefix_pids else None))
    return pd.concat(chunks, ignore_index=True)
