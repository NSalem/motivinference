"""Behaviour statistics outside the pre-registered pipeline (one experiment or the pooled sample).

See :func:`motinf.exploratory.behavior_stats.run_behavior_stats` for the outputs, written
to ``results/exploratory/<exp>/behavior/stats/behav_extra/params/``.

Usage (from repo root)::

    python scripts/exploratory/stats/run_behavior_stats.py --exp-name merged
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

from motinf.exploratory.behavior_stats import run_behavior_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Behaviour statistics outside the pre-registered pipeline.")
    parser.add_argument(
        "--exp-name",
        default="merged",
        help="exp1, exp2 or merged (pools exp1 and exp2).",
    )
    return parser.parse_args()


def main() -> None:
    run_behavior_stats(ROOT, parse_args().exp_name)


if __name__ == "__main__":
    main()
