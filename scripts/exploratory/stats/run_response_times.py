"""Response-time analyses (Figure 5, Supplementary Tables 7-10).

Median response times per condition, the fast / slow split of the noise in both tasks, the response time as a
function of the evidence, and the correlations between incentive effects on response time and on noise. See
``motinf.exploratory.rt_analysis`` (and ``rt_regression``) for the details.

Inputs: ``data/processed/<exp>/data_clean_{est,inf}.csv``, ``results/prereg/<exp>/behavior/data_mean_inc.csv``,
``results/prereg/<exp>/model_fit/fit_noise_per_cond.npz`` and
``results/exploratory/<exp>/modeling/fit_files/fit_inf_rt_split.npz`` (``fit_inf_per_cond_rt_split.py``);
``--exp-name merged`` pools both experiments.
Outputs: ``results/exploratory/<exp>/behavior/stats/rt_analysis/*.csv`` (read by ``paper_tables.py`` and
``scripts/exploratory/figures/fig5.py``).

Usage (from the repository root)::

    python scripts/exploratory/stats/run_response_times.py --exp-name merged
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

from motinf.exploratory.rt_analysis import run_response_times


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RT incentive analyses for one experiment or both pooled.")
    parser.add_argument(
        "--exp-name",
        default="exp1",
        help="exp1, exp2 or merged (pools exp1 and exp2; the output folder has this name).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_response_times(exp_name=args.exp_name, root=ROOT)


if __name__ == "__main__":
    main()
