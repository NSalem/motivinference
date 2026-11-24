"""Participant flow and demographics of the final samples (Methods: Participants, Exclusions).

For each experiment, the flow from recruitment to the final sample:

1. Prolific records: participants who did not complete session 1 or failed its attention check (replaced),
   completed session 1, did not complete session 2, completed both sessions. These counts come from the
   Prolific submission records, which are not distributed because they hold Prolific IDs; their totals are
   in ``data/participant_flow_prolific.csv``.
2. Trial files: participants with data from both sessions, and those whose file for one session is missing.
3. Quality checks of the pre-registered cleaning: participants checked, excluded, and failing each criterion
   (a participant can fail several).
4. Final sample: participants in the cleaned data, with their sex and age (questionnaire of session 2).

The script checks that the counts add up (for example, final = checked - excluded) and stops if they do not.

Inputs:
    data/participant_flow_prolific.csv
    data/raw/<exp>/trials/<participant>/<participant>_sess_<n>_<date>.csv (file names only)
    data/raw/<exp>/questionnaire/questionnaire_data.csv (columns participant, age, sex)
    results/prereg/<exp>/qc/participant_qc_{categ,estim}_simplified.csv
    data/processed/<exp>/data_clean_inf.csv

Outputs (results/exploratory/merged/sample_description/):
    participant_flow.csv     one row per experiment and step, with the source of each count
    exclusion_criteria.csv   participants failing each quality criterion
    demographics.csv         sex counts and age (mean, SD with n - 1, range) of the final samples
    sample_description.txt   the same numbers written as in the Methods

Usage (from the repository root):
    python scripts/exploratory/stats/run_sample_description.py
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
EXPS = ("exp1", "exp2")
OUT = ROOT / "results/exploratory/merged/sample_description"

# Quality criteria of the pre-registered cleaning (src/motinf/prereg/cleaning_core.py), in the order of the
# Methods: label, flag columns (a criterion checked in both tasks has a flag in each QC table).
CRITERIA = [
    ("RT out of range or missed on >= 20% of trials (either task)", ["fail_slowfast_anytask", "fail_slowfast_anytask_estim"]),
    ("RT SD < 0.1 s (either task)", ["fail_rt_sd_low", "fail_rt_sd_low_estim"]),
    ("Stimulus Reproduction: error SD > 30 deg", ["fail_sd_dev_noisy_gt_thresh"]),
    ("Stimulus Reproduction: response SD < 10 deg", ["fail_sd_resp_low"]),
    ("Stimulus Reproduction: mean error > 4 robust SD", ["fail_bias_estim"]),
    ("Stimulus Reproduction: log error SD > 4 robust SD", ["fail_noise_estim"]),
    ("Sequence Categorization: accuracy above the ideal observer", ["fail_above_opt_any_inc_binom"]),
    ("Sequence Categorization: not above chance on the 10% strongest trials", ["fail_easy_top_not_above_chance"]),
    ("Sequence Categorization: psychometric slope <= 0", ["fail_slope_leq_0"]),
    ("Sequence Categorization: same category on >= 90% of trials", ["fail_same_dir"]),
    ("Sequence Categorization: probit bias > 4 robust SD", ["fail_bias_any_inc"]),
    ("Sequence Categorization: probit noise > 4 robust SD", ["fail_noise_any_inc"]),
]
TRIAL_FILE = re.compile(r"^(S\d+)_sess_([12])_")


def session_files(exp: str) -> pd.DataFrame:
    """Number of trial files per participant and session (one row per participant)."""
    counts: dict[str, Counter] = {}
    for f in sorted((ROOT / "data/raw" / exp / "trials").glob("S*/*.csv")):
        m = TRIAL_FILE.match(f.name)
        if m is None:
            raise ValueError(f"unexpected trial file name: {f.name}")
        counts.setdefault(m.group(1), Counter())[int(m.group(2))] += 1
    return pd.DataFrame(
        [{"participant": p, "files_s1": c[1], "files_s2": c[2]} for p, c in sorted(counts.items())]
    )


def quality_checks(exp: str) -> pd.DataFrame:
    qc = ROOT / "results/prereg" / exp / "qc"
    categ = pd.read_csv(qc / "participant_qc_categ_simplified.csv")
    estim = pd.read_csv(qc / "participant_qc_estim_simplified.csv")
    return categ.merge(estim, on="participant", how="outer", suffixes=("", "_estim"), validate="1:1")


def describe(exp: str, prolific: pd.DataFrame) -> tuple[list[dict], list[dict], dict]:
    pro = prolific[prolific["experiment"] == exp].set_index("step")["n"]
    files = session_files(exp)
    both = files[(files.files_s1 > 0) & (files.files_s2 > 0)]
    qc = quality_checks(exp)
    final = pd.read_csv(ROOT / "data/processed" / exp / "data_clean_inf.csv", usecols=["participant"])
    final = sorted(final["participant"].unique())

    excluded = qc["exclude_cat"].astype(bool) | qc["exclude_est"].astype(bool)
    flags = {label: qc[cols].astype(bool).any(axis=1) for label, cols in CRITERIA}
    any_flag = pd.concat(flags, axis=1).any(axis=1)

    # The counts must add up.
    assert pro["session1_completed"] - pro["session2_not_completed"] == pro["both_sessions_completed"], exp
    assert set(qc["participant"]) == set(both["participant"]), f"{exp}: QC set != participants with both sessions"
    assert (any_flag == excluded).all(), f"{exp}: the criteria do not reproduce the exclusions"
    assert sorted(qc.loc[~excluded, "participant"]) == final, f"{exp}: final sample != QC survivors"
    assert len(both) <= pro["both_sessions_completed"], exp

    def row(step: str, n: int, source: str) -> dict:
        return {"experiment": exp, "step": step, "n": int(n), "source": source}

    flow = [
        row("session1_not_completed", pro["session1_not_completed"], "Prolific records"),
        row("session1_failed_attention_check", pro["session1_failed_attention_check"], "Prolific records"),
        row("session1_completed", pro["session1_completed"], "Prolific records"),
        row("session2_not_completed", pro["session2_not_completed"], "Prolific records"),
        row("both_sessions_completed", pro["both_sessions_completed"], "Prolific records"),
        row("participants_with_trial_files", len(files), "trial files"),
        row("both_sessions_in_trial_files", len(both), "trial files"),
        row("only_session1_file", (files.files_s2 == 0).sum(), "trial files"),
        row("only_session2_file", (files.files_s1 == 0).sum(), "trial files"),
        row("session_file_duplicated", ((files.files_s1 > 1) | (files.files_s2 > 1)).sum(),
            "trial files (the cleaning uses the earliest file)"),
        row("both_sessions_completed_but_a_file_missing", pro["both_sessions_completed"] - len(both),
            "Prolific records - trial files"),
        row("quality_checked", len(qc), "quality checks"),
        row("excluded_by_quality_checks", excluded.sum(), "quality checks"),
        row("final_sample", len(final), "cleaned data"),
    ]
    criteria = [
        {"experiment": exp, "criterion": label, "flags": " | ".join(cols), "n_participants": int(flags[label].sum())}
        for label, cols in CRITERIA
    ]

    q = pd.read_csv(ROOT / "data/raw" / exp / "questionnaire/questionnaire_data.csv", usecols=["participant", "age", "sex"])
    q = q[q["participant"].isin(final)]
    assert (q.groupby("participant")[["age", "sex"]].nunique() == 1).all().all(), f"{exp}: inconsistent duplicates"
    q = q.drop_duplicates("participant")
    assert len(q) == len(final), f"{exp}: participants of the final sample without a questionnaire"
    sex = q["sex"].value_counts()
    demo = {
        "experiment": exp,
        "n": len(q),
        "n_male": int(sex.get("Male", 0)),
        "n_female": int(sex.get("Female", 0)),
        "n_other_or_missing": int(len(q) - sex.get("Male", 0) - sex.get("Female", 0)),
        "age_mean": q["age"].mean(),
        "age_sd": q["age"].std(ddof=1),
        "age_min": int(q["age"].min()),
        "age_max": int(q["age"].max()),
    }
    return flow, criteria, demo


def methods_text(flow: pd.DataFrame, criteria: pd.DataFrame, demo: pd.DataFrame) -> str:
    lines = []
    for exp, label in zip(EXPS, ("Experiment 1", "Experiment 2")):
        n = flow[flow.experiment == exp].set_index("step")["n"]
        d = demo[demo.experiment == exp].iloc[0]
        c = criteria[(criteria.experiment == exp) & (criteria.n_participants > 0)]
        lines += [
            f"{label}",
            f"  Session 1: {n.session1_not_completed} did not complete it and "
            f"{n.session1_failed_attention_check} failed the attention check (both replaced); "
            f"{n.session1_completed} completed it.",
            f"  Session 2: {n.session2_not_completed} did not complete it; "
            f"{n.both_sessions_completed} completed both sessions.",
            f"  Trial files: {n.participants_with_trial_files} participants, {n.both_sessions_in_trial_files} with both "
            f"sessions, {n.only_session1_file} with session 1 only, {n.only_session2_file} with session 2 only; "
            f"{n.session_file_duplicated} with a session saved twice. Of the {n.both_sessions_completed} who completed "
            f"both sessions, {n.both_sessions_completed_but_a_file_missing} lack the file of one session.",
            f"  Quality checks: {n.excluded_by_quality_checks} of {n.quality_checked} excluded "
            f"({100 * n.excluded_by_quality_checks / n.quality_checked:.1f}%), N = {n.final_sample}.",
            "  Criteria failed (a participant can fail several): "
            + "; ".join(f"{r.criterion}: {r.n_participants}" for r in c.itertuples()) + ".",
            f"  Final sample: {d.n} participants ({d.n_male} male, {d.n_female} female; mean age = {d.age_mean:.2f} "
            f"years, SD = {d.age_sd:.2f}, range = {d.age_min}–{d.age_max}).",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    prolific = pd.read_csv(ROOT / "data/participant_flow_prolific.csv")
    flow, criteria, demo = [], [], []
    for exp in EXPS:
        f, c, d = describe(exp, prolific)
        flow += f
        criteria += c
        demo.append(d)
    flow, criteria, demo = pd.DataFrame(flow), pd.DataFrame(criteria), pd.DataFrame(demo)
    OUT.mkdir(parents=True, exist_ok=True)
    flow.to_csv(OUT / "participant_flow.csv", index=False)
    criteria.to_csv(OUT / "exclusion_criteria.csv", index=False)
    demo.to_csv(OUT / "demographics.csv", index=False)
    text = methods_text(flow, criteria, demo)
    (OUT / "sample_description.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"Written to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
