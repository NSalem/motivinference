"""Preregistered behavioral analyses entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from motinf.prereg.behavior import run_behavior_main_prereg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preregistered behavioral analyses for one experiment.")
    parser.add_argument("--exp-name", default="exp1", help="Experiment subfolder under data/processed/.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_behavior_main_prereg(exp_name=args.exp_name)


if __name__ == "__main__":
    main()
