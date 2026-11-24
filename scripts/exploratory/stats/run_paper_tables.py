"""Every test statistic of the paper's tables and text, in one file (Tables 1-3, Supplementary Tables 1-10).

Computed by ``motinf.exploratory.paper_tables`` from the participant-level outputs of the other steps of the stats
stage (run them first); its docstring maps each table to its keys and states the conventions (contrasts, dz with
exact CIs, Holm correction, Bayes factors, exact Greenhouse-Geisser correction).

Output: ``results/exploratory/merged/paper_tables/paper_tables.json``.

Usage (from the repository root)::

    python scripts/exploratory/stats/run_paper_tables.py
"""

from pathlib import Path

from motinf.exploratory.paper_tables import run_paper_tables

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    print(run_paper_tables(root))
