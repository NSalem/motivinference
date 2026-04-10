"""Preregistered cleaning orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .cleaning_core import QCConfig, quality_check_exclusion_separate_tasks

from .aggregation import aggregate_data_prereg


@dataclass
class QCConfigPrereg:
    """Thin prereg wrapper around QCConfig with explicit conversion."""

    outdir: str

    def to_base(self) -> QCConfig:
        return QCConfig(outdir=self.outdir)


def _count_csv_rows(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8") as f:
        n_lines = sum(1 for _ in f)
    return max(n_lines - 1, 0)


def _build_exclusion_summary(exp_name: str, n_trials_cat_in: int, n_trials_est_in: int, qc_out: dict) -> str:
    data_clean_cat = qc_out["data_clean_cat"]
    data_clean_est = qc_out["data_clean_est"]
    bad_all = qc_out["bad_all"]

    n_trials_cat_out = len(data_clean_cat)
    n_trials_est_out = len(data_clean_est)
    n_trials_cat_removed = n_trials_cat_in - n_trials_cat_out
    n_trials_est_removed = n_trials_est_in - n_trials_est_out
    participant_ids = sorted(bad_all["participant"].astype(str).tolist())
    participant_ids_str = ", ".join(participant_ids) if participant_ids else "(none)"

    lines = [
        f"Exclusion summary for {exp_name}",
        "-" * 48,
        f"Trials removed (categorization): {n_trials_cat_removed} / {n_trials_cat_in}",
        f"Trials removed (estimation):     {n_trials_est_removed} / {n_trials_est_in}",
        f"Trials removed (total):          {n_trials_cat_removed + n_trials_est_removed} / {n_trials_cat_in + n_trials_est_in}",
        "",
        f"Participants removed: {len(participant_ids)}",
        f"Participant IDs: {participant_ids_str}",
    ]
    return "\n".join(lines) + "\n"


def run_cleaning_prereg(exp_name: str = "exp1") -> None:
    """Run prereg cleaning flow for one experiment."""
    raw_folder = Path("data/raw") / exp_name / "trials"
    agg_folder = Path("data/interim") / exp_name
    clean_folder = Path("data/processed") / exp_name
    qc_folder = Path("results/prereg") / exp_name / "qc"
    clean_folder.mkdir(parents=True, exist_ok=True)
    qc_folder.mkdir(parents=True, exist_ok=True)

    aggregate_data_prereg(in_folder=str(raw_folder), out_folder=str(agg_folder))

    cat_path = agg_folder / "all_trials_inf.csv"
    est_path = agg_folder / "all_trials_est.csv"
    qc_cfg = QCConfigPrereg(outdir=str(qc_folder)).to_base()
    qc_out = quality_check_exclusion_separate_tasks(
        df_cat=str(cat_path),
        df_est=str(est_path),
        out_cat_path=str(clean_folder / "data_clean_inf.csv"),
        out_est_path=str(clean_folder / "data_clean_est.csv"),
        config=qc_cfg,
    )

    summary_txt = _build_exclusion_summary(
        exp_name=exp_name,
        n_trials_cat_in=_count_csv_rows(cat_path),
        n_trials_est_in=_count_csv_rows(est_path),
        qc_out=qc_out,
    )
    print(summary_txt, end="")
    (qc_folder / "exclusion_summary.txt").write_text(summary_txt, encoding="utf-8")
