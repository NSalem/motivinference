"""Run the analyses behind the paper, stage by stage.

Stages (run in this order when several are given):

``fit``
    Model fits and simulation studies (slow: days of CPU time in total; the full-model fits and recoveries were
    run on a computing cluster). The pre-registered pipeline per experiment (cleaning, behaviour, single-noise and
    single-modulation fits, model statistics), the full model (BADS then VBMC), the fast/slow RT-split fits, and
    the identifiability and recovery studies.
``stats``
    The participant-level tables, from the trial data and the fits (needs the data bundle): pre-registered behaviour
    and model statistics, bias-noise partition, model predictions and parameters, biases, response times,
    questionnaire regressions, sample description.
``tables``
    The statistics computed from those tables only (no data needed: they are in ``results/``): the tests and group
    model comparisons of ``motinf.exploratory.table_stats``, then every test statistic of the paper and supplement in
    ``paper_tables.json``.
``figures``
    Every figure of the paper and supplement, drawn whole at print size from the tables in ``results/``
    (``scripts/exploratory/figures``).
``demo``
    A short check of the fitting code (about 30 seconds): refits the pre-registered single-noise models to the first
    two participants of Experiment 1, with the seed and settings of the full run, and compares the result with the
    bundled fits (``checks/demo_refit.py``).
``verify``
    Compares the outputs on disk with the recorded ones (``checks/compare_outputs.py``): byte for byte where possible,
    numerically for tables that differ in their bytes; and checks the table of ``PREREG_DEVIATIONS.md``.

From the code repository alone, ``--stage tables figures verify`` recomputes and checks every test statistic and
figure in a few minutes. With the data bundle (see ``data/README.md``), ``--stage stats tables figures verify``
also recomputes the participant-level tables, without refitting. ``PAPER_MAP.md`` lists which step produces which
table, figure or number.

Usage (from the repository root, in the environment of ``environment.yml``)::

    python run_pipeline.py --stage tables figures verify
    python run_pipeline.py --stage stats tables figures verify
    python run_pipeline.py --stage demo
    python run_pipeline.py --stage fit --dry-run
    python run_pipeline.py --stage fit

``data/README.md`` says how closely the ``fit`` stage reproduces each bundled fit.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPS = ("exp1", "exp2")
PREREG = "scripts/prereg"
FITS = "scripts/exploratory/fits"
STATS = "scripts/exploratory/stats"
FIG = "scripts/exploratory/figures"
FIGURES = ("fig1", "fig2", "fig3", "fig4", "fig5", "sfig1", "sfig2", "sfig3", "sfig4", "sfig5", "sfig6", "sfig7")
STAGES = ("fit", "stats", "tables", "figures", "demo", "verify")


def _steps() -> list[tuple[str, str, list[str]]]:
    """(stage, label, [script, *args]) in execution order."""
    s: list[tuple[str, str, list[str]]] = []
    # -- fit ---------------------------------------------------------------------------------------
    for e in EXPS:
        s.append(("fit", f"prereg_pipeline_{e}", [f"{PREREG}/run_prereg_pipeline.py", "--exp-name", e]))
    for e in EXPS:
        s.append(("fit", f"fit_full_model_{e}", [f"{FITS}/fit_full_model.py", "--exp-name", e, "--vbmc-after-bads"]))
        s.append(("fit", f"fit_rt_split_{e}", [f"{FITS}/fit_inf_per_cond_rt_split.py", "--exp-name", e]))
    s += [
        ("fit", "identifiability_single_noise_exp1",
         [f"{FITS}/run_identifiability_single_noise.py", "--exp-name", "exp1", "--n-reps", "20"]),
        ("fit", "identifiability_single_modulation_exp1",
         [f"{FITS}/run_identifiability_single_modulation.py", "--exp-name", "exp1", "--n-reps", "20"]),
        ("fit", "recovery_inference_only_exp1", [f"{FITS}/run_recovery_inference_only.py", "--exp-name", "exp1"]),
        ("fit", "recovery_full_model_merged",
         [f"{FITS}/run_recovery_full_model.py", "--exp-name", "merged", "--bads-nit", "10", "--vbmc-after-bads"]),
    ]
    # -- stats -------------------------------------------------------------------------------------
    for e in EXPS:
        s.append(("stats", f"prereg_behavior_{e}", [f"{PREREG}/run_behavior_main.py", "--exp-name", e]))
        s.append(("stats", f"prereg_model_stats_{e}", [f"{PREREG}/run_model_stats.py", "--exp-name", e]))
    for e in EXPS:
        s.append(("stats", f"variance_partition_{e}", [f"{STATS}/run_variance_partition.py", "--exp-name", e]))
    for e in (*EXPS, "merged"):
        s.append(("stats", f"behavior_stats_{e}", [f"{STATS}/run_behavior_stats.py", "--exp-name", e]))
    s += [
        ("stats", "inference_model_stats_merged", [f"{STATS}/run_inference_model_stats.py", "--exp-name", "merged"]),
        ("stats", "full_model_stats_merged", [f"{STATS}/run_full_model_stats.py", "--exp-name", "merged"]),
    ]
    for e in (*EXPS, "merged"):
        s.append(("stats", f"bias_stats_{e}", [f"{STATS}/run_bias_stats.py", "--exp-name", e]))
    s += [
        ("stats", "questionnaire_merged", [f"{STATS}/run_questionnaire_model_correlations.py", "--exp-name", "merged"]),
        ("stats", "response_times_merged", [f"{STATS}/run_response_times.py", "--exp-name", "merged"]),
        ("stats", "full_vs_inference_only_bmc", [f"{STATS}/run_full_vs_inference_only_bmc.py"]),
        ("stats", "sample_description", [f"{STATS}/run_sample_description.py"]),
        # -- tables: from participant-level tables only ------------------------------------------------
        ("tables", "table_stats", [f"{STATS}/run_table_stats.py"]),
        ("tables", "paper_tables", [f"{STATS}/run_paper_tables.py"]),
    ]
    # -- figures (labels fig1 ... sfig7) -------------------------------------------------------------
    s += [("figures", name, [f"{FIG}/{name}.py"]) for name in FIGURES]
    # -- demo and verify -----------------------------------------------------------------------------
    s += [
        ("demo", "demo_refit", ["checks/demo_refit.py"]),
        ("verify", "compare_outputs", ["checks/compare_outputs.py"]),
        ("verify", "prereg_deviations", ["checks/prereg_deviations.py", "--check"]),
    ]
    return s


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--stage", nargs="+", choices=STAGES, required=True, help="Stage(s) to run, in pipeline order.")
    p.add_argument("--skip", nargs="*", default=[], help="Step labels to skip (see --dry-run for the labels).")
    p.add_argument("--dry-run", action="store_true", help="Print the commands without running them.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    env = dict(os.environ)
    src = str(ROOT / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("SOURCE_DATE_EPOCH", "0")  # no creation dates in SVG/PDF metadata
    env.setdefault("PYTHONHASHSEED", "0")

    steps = [st for st in _steps() if st[0] in args.stage and st[1] not in set(args.skip)]
    if not args.dry_run and set(args.stage) & {"fit", "stats", "tables", "figures"}:
        # keep the recorded versions of the outputs before they are overwritten, for --stage verify without git
        subprocess.run([sys.executable, "checks/output_manifest.py", "--snapshot"], cwd=ROOT, env=env, check=True)
    for stage, label, cmd in steps:
        print(f"[{stage}] {label}: {' '.join([sys.executable, *cmd])}", flush=True)
        if args.dry_run:
            continue
        t0 = time.time()
        subprocess.run([sys.executable, *cmd], cwd=ROOT, env=env, check=True)
        print(f"    done in {time.time() - t0:.0f} s", flush=True)
    print("Pipeline finished." if not args.dry_run else f"{len(steps)} steps.")


if __name__ == "__main__":
    main()
