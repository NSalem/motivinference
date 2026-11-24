"""The pre-registered pipeline is the frozen code plus the deviations of PREREG_DEVIATIONS.md, and Table 1 is its output.

1. Every file in ``src/motinf/prereg`` and ``scripts/prereg`` has the hash it had at the git tag ``prereg``
   (``checks/prereg_frozen.sha256``), except the files the deviations change; ``rm_anova_exact.py`` is the only added
   file; each changed or added file is named in ``PREREG_DEVIATIONS.md``.
2. The pre-registered tier imports nothing from the exploratory or extras tiers.
3. Table 1 (``paper_tables.json`` -> ``table1``) equals what the pre-registered code writes or computes: its exact-GG
   ANOVAs and its one-tailed t-tests.
4. The table of ``PREREG_DEVIATIONS.md`` is current (``checks/prereg_deviations.py --check``).

Tests 3 and 4 need the data bundle and are skipped without it.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]

FROZEN = ROOT / "checks" / "prereg_frozen.sha256"
DOC = ROOT / "PREREG_DEVIATIONS.md"
PT = ROOT / "results" / "exploratory" / "merged" / "paper_tables" / "paper_tables.json"
TIER = ("src/motinf/prereg", "scripts/prereg")
CHANGED = {  # deviation 1, deviation 2, documentation
    "src/motinf/prereg/model.py",
    "src/motinf/prereg/behavior.py",
    "scripts/prereg/run_model_stats.py",
    "scripts/prereg/README.md",
}
ADDED = {"src/motinf/prereg/rm_anova_exact.py"}
EXPS = ("exp1", "exp2")

needs_bundle = pytest.mark.skipif(
    not (PT.exists() and (ROOT / "data" / "processed" / "exp1" / "data_clean_inf.csv").exists()),
    reason="data bundle not unpacked",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _tier_files() -> dict[str, Path]:
    return {p.relative_to(ROOT).as_posix(): p for d in TIER for p in (ROOT / d).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts}


def _close(a: float, b: float, rel: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= rel * max(1.0, abs(float(b)))


def test_frozen_code_changed_only_where_documented() -> None:
    frozen = {}
    for line in FROZEN.read_text(encoding="utf-8").splitlines():
        digest, path = line.split(maxsplit=1)
        frozen[path] = digest
    files = _tier_files()
    assert set(files) == set(frozen) | ADDED, (
        f"unexpected: {sorted(set(files) - set(frozen) - ADDED)}, missing: {sorted(set(frozen) - set(files))}")
    for path, digest in frozen.items():
        if path in CHANGED:
            assert _sha(files[path]) != digest, f"{path} is listed as changed but equals the frozen version"
        else:
            assert _sha(files[path]) == digest, f"{path} differs from the frozen code but is not a documented deviation"
    doc = DOC.read_text(encoding="utf-8")
    for path in CHANGED | ADDED:
        assert f"`{path}`" in doc, f"{path} is not named in PREREG_DEVIATIONS.md"


def test_prereg_tier_is_self_contained() -> None:
    bad = []
    for rel, path in _tier_files().items():
        if path.suffix != ".py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                [node.module] if isinstance(node, ast.ImportFrom) and node.module and not node.level else [])
            bad += [f"{rel}: {m}" for m in mods if m.startswith(("motinf.exploratory", "motinf.extras"))]
    assert not bad, "the pre-registered tier imports other tiers:\n" + "\n".join(bad)


@needs_bundle
@pytest.mark.parametrize("exp", EXPS)
def test_table1_is_the_preregistered_pipelines_output(exp: str) -> None:
    from motinf.prereg import behavior as B

    t1 = json.loads(PT.read_text(encoding="utf-8"))["table1"][exp]
    res = ROOT / "results" / "prereg" / exp
    ex = pd.read_csv(res / "behavior" / "rm_anova_exact_gg.csv")

    def row(dv: str, source: str) -> pd.Series:
        (i,) = ex.index[(ex["dv"] == dv) & (ex["Source"] == source)]
        return ex.loc[i]

    for r, t in ((row("sensitivity", "incentive"), t1["sensitivity"]["omnibus"]),        # H1
                 (row("choice_var_log", "seqlen"), t1["choice_var"]["seqlen"]),          # H2 ANOVA
                 (row("choice_var_log", "incentive"), t1["choice_var"]["incentive"]),
                 (row("choice_var_log", "seqlen * incentive"), t1["choice_var"]["seqlen_x_incentive"]),  # H2
                 (row("estim_sd", "incentive"), t1["estim_sd"]["omnibus"])):             # pre-specified
        assert (int(r["ddof1"]), int(r["ddof2"])) == (t["df1"], t["df2"])
        assert _close(r["F"], t["F"]) and _close(r["p_GG_corr"], t["p_gg"]) and _close(r["np2"], t["np2"])

    an = pd.read_csv(res / "model_fit" / "stats" / "anova_sd_inf_incentive.csv").set_index("Source").loc["incentive"]
    t = t1["log_sd_inf"]["omnibus"]                                                       # H4A
    assert _close(an["F"], t["F"]) and _close(an["p_GG_exact"], t["p_gg"]) and _close(an["np2"], t["np2"])

    beh = pd.read_csv(res / "behavior" / "data_mean_inc.csv")
    beh["incentive_abs"] = beh["incentive"].abs()
    for dv, alternative in (("sensitivity", "greater"), ("estim_sd", "less")):           # H3, pre-specified
        tt = B._paired_abs_ttest(beh, dv=dv, alternative=alternative).rename(columns=lambda c: c.replace("-", "_")).iloc[0]
        assert _close(tt["T"], t1[dv]["abs"]["t"]) and _close(tt["p_val"], t1[dv]["abs"]["p"])
    h4b = pd.read_csv(res / "model_fit" / "stats" / "ttest_log_sd_inf_abs_1_vs_0.csv")
    h4b = h4b.rename(columns=lambda c: c.replace("-", "_")).iloc[0]                      # H4B
    assert _close(h4b["T"], t1["log_sd_inf"]["abs"]["t"]) and _close(h4b["p_val"], t1["log_sd_inf"]["abs"]["p"])


@needs_bundle
def test_deviation_table_is_current() -> None:
    pytest.importorskip("groupBMC")
    if not (ROOT / "results" / "prereg" / "exp1" / "model_fit" / "as_registered" / "fit_noise_per_cond.npz").exists():
        pytest.skip("fits as registered not in the data bundle")
    r = subprocess.run([sys.executable, str(ROOT / "checks" / "prereg_deviations.py"), "--check"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
