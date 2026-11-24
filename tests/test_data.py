"""The trial data match the design and the exclusions described in the Methods (skipped without the data bundle).

``data/interim/<exp>/all_trials_*.csv``: every trial of every participant who completed both sessions;
``data/processed/<exp>/data_clean_*.csv``: the trials and participants retained by the quality checks
(``motinf.prereg.cleaning``). The numbers below are the ones printed in the Methods.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
ANGLES = [f"seqang_noisy_{i}" for i in range(1, 13)]
COLS = {"inf": ["participant", "session", "trialN", "incentive", "seqlen", "true_cat", "rt", *ANGLES],
        "est": ["participant", "session", "trialN", "incentive", "estimang_noisy", "gaborimg_estim", "rt"]}
TRIALS_PER_SESSION = {"inf": 432, "est": 72}  # Sequence Categorization, Stimulus Reproduction
REPEAT_LAG = {"inf": 216, "est": 36}  # the second half of each task in a session repeats the first
ENTERED_QC = {"exp1": 100, "exp2": 148}
N_FINAL = {"exp1": 84, "exp2": 132}
RT_REMOVED = {("exp1", "inf"): 397, ("exp1", "est"): 51, ("exp2", "inf"): 217, ("exp2", "est"): 51}
EXPS = ("exp1", "exp2")
TASKS = ("inf", "est")


@lru_cache(maxsize=None)
def _load(stage: str, exp: str, task: str) -> pd.DataFrame:
    name = f"all_trials_{task}.csv" if stage == "interim" else f"data_clean_{task}.csv"
    path = ROOT / "data" / stage / exp / name
    if not path.is_file():
        pytest.skip(f"{path.relative_to(ROOT)} not found (data bundle not unpacked; see data/README.md)")
    df = pd.read_csv(path, usecols=COLS[task])
    df["pos"] = df.groupby(["participant", "session"])["trialN"].rank(method="first").astype(int) - 1
    return df


def _retained(exp: str, task: str) -> pd.DataFrame:
    """Interim trials of the participants kept by the quality checks."""
    keep = set(_load("processed", exp, task)["participant"])
    df = _load("interim", exp, task)
    return df[df["participant"].isin(keep)]


@pytest.mark.parametrize("exp", EXPS)
def test_sample_sizes(exp: str) -> None:
    for task in TASKS:
        assert _load("interim", exp, task)["participant"].nunique() == ENTERED_QC[exp]
        assert _load("processed", exp, task)["participant"].nunique() == N_FINAL[exp]
    assert set(_load("processed", exp, "inf")["participant"]) == set(_load("processed", exp, "est")["participant"])


@pytest.mark.parametrize("exp", EXPS)
@pytest.mark.parametrize("task", TASKS)
def test_two_full_sessions_per_participant(exp: str, task: str) -> None:
    df = _load("interim", exp, task)
    sizes = df.groupby(["participant", "session"]).size()
    assert sorted(df["session"].unique()) == [1, 2]
    assert (sizes == TRIALS_PER_SESSION[task]).all() and len(sizes) == 2 * ENTERED_QC[exp]


@pytest.mark.parametrize("exp", EXPS)
@pytest.mark.parametrize("task", TASKS)
def test_rt_cuts_remove_exactly_the_out_of_range_trials(exp: str, task: str) -> None:
    """Trials below 0.2 s or above 10 s are removed (Experiment 2 had deadlines, so none exceeds 10 s there)."""
    before, after = _retained(exp, task), _load("processed", exp, task)
    assert len(before) == 2 * TRIALS_PER_SESSION[task] * N_FINAL[exp]
    in_range = before[(before["rt"] >= 0.2) & (before["rt"] <= 10)]
    key = ["participant", "session", "trialN"]
    assert len(before) - len(after) == RT_REMOVED[(exp, task)]
    assert in_range[key].sort_values(key).reset_index(drop=True).equals(after[key].sort_values(key).reset_index(drop=True))


@pytest.mark.parametrize("exp", EXPS)
def test_every_participant_has_every_condition(exp: str) -> None:
    inf = _retained(exp, "inf")
    assert (inf.groupby(["participant", "incentive", "seqlen"]).size() == 96).all()  # 864 / (3 incentives × 3 lengths)
    assert inf.groupby("participant").apply(lambda d: d.groupby(["incentive", "seqlen"]).ngroups).eq(9).all()
    est = _retained(exp, "est")
    assert (est.groupby(["participant", "incentive"]).size() == 48).all()
    clean = _load("processed", exp, "inf")
    assert clean.groupby("participant").apply(lambda d: d.groupby(["incentive", "seqlen"]).ngroups).eq(9).all()


@pytest.mark.parametrize("exp", EXPS)
@pytest.mark.parametrize("task", TASKS)
def test_each_trial_is_repeated_later_in_its_session(exp: str, task: str) -> None:
    """Within each session, trial k + 216 (Sequence Categorization) or k + 36 (Stimulus Reproduction) repeats trial k:
    the same stimuli and the same incentive."""
    df = _load("interim", exp, task)
    lag = REPEAT_LAG[task]
    first = df[df["pos"] < lag]
    second = df[df["pos"] >= lag].assign(pos=lambda d: d["pos"] - lag)
    pairs = first.merge(second, on=["participant", "session", "pos"], suffixes=("", "_rep"))
    assert len(pairs) == len(first) == len(df) // 2
    same = ["incentive"] + (["seqlen", "true_cat"] if task == "inf" else ["gaborimg_estim"])
    for c in same:
        assert (pairs[c] == pairs[c + "_rep"]).all(), c
    angles = ANGLES if task == "inf" else ["estimang_noisy"]
    a, b = pairs[angles].to_numpy(), pairs[[c + "_rep" for c in angles]].to_numpy()
    assert np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(a[~np.isnan(a)], b[~np.isnan(b)], rtol=0, atol=1e-12)
