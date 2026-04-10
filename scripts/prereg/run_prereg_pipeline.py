"""Run the full preregistered pipeline for one experiment (cleaning through model stats)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREREG = ROOT / "scripts" / "prereg"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run cleaning, behavior, both model fits, then model statistics."
    )
    p.add_argument("--exp-name", default="exp1", help="Experiment id (data/raw, data/processed, results/prereg).")
    p.add_argument(
        "--max-nsubs",
        type=int,
        default=0,
        help="Forward to fit scripts: cap participants (0 = all).",
    )
    p.add_argument(
        "--bads-nit",
        type=int,
        default=10,
        help="Forward to fit scripts: BADS restarts per fit.",
    )
    return p.parse_args()


def _run_step(label: str, argv: list[str]) -> None:
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}\n", flush=True)
    subprocess.run([sys.executable, *argv], cwd=str(ROOT), check=True)


def main() -> None:
    args = parse_args()
    exp = args.exp_name
    fit_extra = []
    if args.max_nsubs > 0:
        fit_extra.extend(["--max-nsubs", str(args.max_nsubs)])
    fit_extra.extend(["--bads-nit", str(args.bads_nit)])

    _run_step(
        "1/5 Cleaning (aggregate + QC)",
        [str(PREREG / "run_cleaning.py"), "--exp-name", exp],
    )
    _run_step(
        "2/5 Behavioral statistics",
        [str(PREREG / "run_behavior_main.py"), "--exp-name", exp],
    )
    _run_step(
        "3/5 Per-condition model fits",
        [str(PREREG / "fit_noise_per_cond.py"), "--exp-name", exp, *fit_extra],
    )
    _run_step(
        "4/5 Incentive-channel model fits",
        [str(PREREG / "fit_incentive_effects.py"), "--exp-name", exp, *fit_extra],
    )
    _run_step(
        "5/5 Model-fit statistics (Group BMC, tables)",
        [str(PREREG / "run_model_stats.py"), "--exp-name", exp],
    )
    print(f"\nPipeline finished for exp_name={exp!r}.\n", flush=True)


if __name__ == "__main__":
    main()
