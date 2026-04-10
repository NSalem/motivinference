"""Preregistered cleaning entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from motinf.prereg.cleaning import run_cleaning_prereg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preregistered aggregate + cleaning for one experiment.")
    parser.add_argument("--exp-name", default="exp1", help="Experiment subfolder inside data/raw/ (e.g., exp1, exp2).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_cleaning_prereg(exp_name=args.exp_name)


if __name__ == "__main__":
    main()
