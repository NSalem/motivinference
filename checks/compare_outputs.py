"""Compare the outputs on disk with the recorded ones (the ``verify`` stage of ``run_pipeline.py``).

1. Byte check: every file recorded in ``checks/expected_outputs.sha256`` is present and has the recorded SHA-256.
   On the machine and in the environment that wrote the record, a full run reproduces every file byte for byte.
2. Files whose bytes differ are compared by content with the recorded version: from ``--reference`` (a copy of the
   repository as distributed), else the copy that ``run_pipeline.py`` kept before overwriting it
   (``build/recorded_outputs/``, by SHA-256), else the last git commit:

   - tables and text (``.csv``, ``.json``, ``.txt``, ``.md``): the text around the numbers must be identical and
     every number equal within a relative tolerance (default 1e-6, far below the printed precision; values closer
     than 1e-12 count as equal). Line endings are ignored.
   - arrays (``.npz``): same keys and shapes, numbers within the same tolerance.
   - figures (``.svg``, ``.png``, ``.pdf``): listed but not compared, because their bytes depend on the fonts and
     library versions; compare the PNG previews with the committed ones by eye.

Prints one line per file that differs, then PASS or FAIL. Exit code 1 if a recorded file is missing or a table,
number or array differs beyond the tolerance.

Usage (from the repository root):
    python checks/compare_outputs.py
    python checks/compare_outputs.py --reference <unpacked copy of the repository> --rtol 1e-9
"""

from __future__ import annotations

import argparse
import io
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "checks"))
from output_manifest import SNAPSHOT, bundle_unpacked, current, in_bundle, read_manifest  # noqa: E402

TEXT = {".csv", ".json", ".txt", ".md"}
FIGURE = {".svg", ".png", ".pdf"}
ATOL = 1e-12
NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def rel_diff(a: float, b: float) -> float:
    if math.isnan(a) or math.isnan(b):
        return 0.0 if math.isnan(a) and math.isnan(b) else math.inf
    if a == b or abs(a - b) <= ATOL:
        return 0.0
    return abs(a - b) / max(abs(a), abs(b))


def compare_text(new: bytes, ref: bytes) -> tuple[float, str]:
    """Largest relative difference between the numbers of two texts (inf if the text around them differs)."""
    new_lines = new.decode("utf-8").replace("\r\n", "\n").split("\n")
    ref_lines = ref.decode("utf-8").replace("\r\n", "\n").split("\n")
    if len(new_lines) != len(ref_lines):
        return math.inf, f"{len(new_lines)} lines instead of {len(ref_lines)}"
    worst, where = 0.0, ""
    for i, (a, b) in enumerate(zip(new_lines, ref_lines), start=1):
        if a == b:
            continue
        if NUMBER.sub("#", a) != NUMBER.sub("#", b):
            return math.inf, f"line {i}: {a[:80]!r} vs {b[:80]!r}"
        for x, y in zip(NUMBER.findall(a), NUMBER.findall(b)):
            d = rel_diff(float(x), float(y))
            if d > worst:
                worst, where = d, f"line {i}: {x} vs {y}"
    return worst, where


def compare_values(a, b) -> float:
    """Largest relative difference between two loaded values (arrays, dicts, sequences, scalars)."""
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.shape != b.shape:
            return math.inf
        if a.dtype == object or b.dtype == object:
            return max((compare_values(x, y) for x, y in zip(a.ravel(), b.ravel())), default=0.0)
        if a.dtype.kind in "biuf" and b.dtype.kind in "biuf":
            return max((rel_diff(float(x), float(y)) for x, y in zip(a.ravel(), b.ravel())), default=0.0)
        return 0.0 if np.array_equal(a, b) else math.inf
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return math.inf
        return max((compare_values(a[k], b[k]) for k in a), default=0.0)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return math.inf
        return max((compare_values(x, y) for x, y in zip(a, b)), default=0.0)
    if isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)) \
            and not isinstance(a, bool) and not isinstance(b, bool):
        return rel_diff(float(a), float(b))
    return 0.0 if a == b else math.inf


def compare_npz(new: bytes, ref: bytes) -> tuple[float, str]:
    with np.load(io.BytesIO(new), allow_pickle=True) as a, np.load(io.BytesIO(ref), allow_pickle=True) as b:
        if set(a.files) != set(b.files):
            return math.inf, f"keys {sorted(set(a.files) ^ set(b.files))} differ"
        worst, where = 0.0, ""
        for k in a.files:
            d = compare_values(a[k], b[k])
            if d > worst:
                worst, where = d, f"key {k}"
        return worst, where


def _is_git_root() -> bool:
    r = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT, capture_output=True, text=True)
    return r.returncode == 0 and Path(r.stdout.strip()).resolve() == ROOT.resolve()


def recorded_copy(path: str, digest: str, reference: Path | None) -> bytes | None:
    """The recorded version of an output: from --reference, else the copy kept before the run overwrote it
    (``output_manifest.py --snapshot``), else the last commit when this folder is a git repository."""
    if reference is not None:
        p = reference / path
        return p.read_bytes() if p.is_file() else None
    if (SNAPSHOT / digest).is_file():
        return (SNAPSHOT / digest).read_bytes()
    if _is_git_root():
        r = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=ROOT, capture_output=True)
        return r.stdout if r.returncode == 0 else None
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reference", type=Path, help="copy of the repository holding the recorded outputs "
                    "(default: the last commit of this git repository)")
    ap.add_argument("--rtol", type=float, default=1e-6, help="relative tolerance for numbers (default 1e-6)")
    args = ap.parse_args()

    recorded, now = read_manifest(), current()
    differ = sorted(p for p in recorded if p in now and now[p] != recorded[p])
    missing = sorted(p for p in recorded if p not in now)
    elsewhere = [p for p in missing if in_bundle(p) and not bundle_unpacked()]  # the fits, in the data bundle
    missing = [p for p in missing if p not in elsewhere]
    new = sorted(p for p in now if p not in recorded)
    print(f"Byte check ({len(recorded)} recorded files): "
          f"{len(recorded) - len(differ) - len(missing) - len(elsewhere)} identical, {len(differ)} differ, "
          f"{len(missing)} missing, {len(new)} not recorded"
          + (f"; {len(elsewhere)} model fits are in the data bundle (not unpacked, not checked)" if elsewhere else ""))

    failed = list(missing)
    for p in missing:
        print(f"  MISSING    {p}")
    for p in new:
        print(f"  NEW        {p} (not in the record; not checked)")
    source = f"--reference {args.reference}" if args.reference else "the recorded versions"
    if differ:
        print(f"Files whose bytes differ, compared with {source}:")
    for p in differ:
        suffix = Path(p).suffix
        if suffix in FIGURE:
            print(f"  FIGURE     {p} (not compared)")
            continue
        ref = recorded_copy(p, recorded[p], args.reference)
        if ref is None:
            print(f"  NO COPY    {p} (no recorded version to compare with)")
            failed.append(p)
            continue
        new_bytes = (ROOT / p).read_bytes()
        if suffix == ".npz":
            worst, where = compare_npz(new_bytes, ref)
        elif suffix in TEXT:
            worst, where = compare_text(new_bytes, ref)
        else:
            worst, where = (0.0, "") if new_bytes == ref else (math.inf, "binary file")
        if worst <= args.rtol:
            print(f"  EQUAL      {p} (largest relative difference {worst:.1e})")
        else:
            print(f"  DIFFERENT  {p} (largest relative difference {worst:.1e}; {where})")
            failed.append(p)

    if failed:
        print(f"FAIL: {len(failed)} file(s) missing or different beyond the tolerance (relative {args.rtol:g}).")
        return 1
    print(f"PASS: every recorded table, number and array is reproduced (relative tolerance {args.rtol:g}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
