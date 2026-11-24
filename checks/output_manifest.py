"""Golden master of the pipeline's outputs: a SHA-256 manifest of every file under ``results/prereg/``,
``results/exploratory/`` and ``plots/exploratory/``.

The manifest (``checks/expected_outputs.sha256``) uses the ``sha256sum`` format, so it can also be checked without
Python: ``sha256sum -c checks/expected_outputs.sha256`` from the repository root.

Usage (from the repository root):

    python checks/output_manifest.py --write    # record the current outputs
    python checks/output_manifest.py --check    # compare the current outputs with the record (exit code 1 if any differ)
    python checks/output_manifest.py --snapshot # keep the recorded versions in build/recorded_outputs/ (run_pipeline.py
                                                # does this before any stage that writes outputs)

The model fits (``.npz``) are recorded too but distributed in the data bundle; without it they are reported, not
counted as missing.

Byte identity is the gate for refactoring on the machine and environment that wrote the record (``environment.yml``,
fonts included).
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "checks" / "expected_outputs.sha256"
TREES = ("results/prereg", "results/exploratory", "plots/exploratory")
SKIP_DIRS = {"__pycache__"}


def output_files() -> list[Path]:
    files = []
    for tree in TREES:
        for p in (ROOT / tree).rglob("*"):
            rel = p.relative_to(ROOT)
            if p.is_file() and not SKIP_DIRS.intersection(rel.parts) and "Zone.Identifier" not in p.name:
                files.append(p)
    return sorted(files, key=lambda p: p.relative_to(ROOT).as_posix())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def current() -> dict[str, str]:
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in output_files()}


def read_manifest() -> dict[str, str]:
    out = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, path = line.split(maxsplit=1)
        out[path] = digest
    return out


SNAPSHOT = ROOT / "build" / "recorded_outputs"  # recorded versions, by SHA-256 (not distributed)


def in_bundle(path: str) -> bool:
    """Recorded outputs distributed in the data bundle rather than with the code: the model fits."""
    return path.endswith(".npz")


def bundle_unpacked() -> bool:
    return (ROOT / "data" / "processed").is_dir()


def snapshot_recorded() -> int:
    """Keep a copy of every output that still has its recorded bytes, under its SHA-256, so that a copy of the release
    without git still has the recorded versions to compare with after a run overwrites them (``compare_outputs.py``).
    Returns the number of files copied."""
    n = 0
    for path, digest in read_manifest().items():
        src, dst = ROOT / path, SNAPSHOT / digest
        if dst.is_file() or not src.is_file() or sha256(src) != digest:
            continue
        SNAPSHOT.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="record the current outputs")
    mode.add_argument("--check", action="store_true", help="compare the current outputs with the record")
    mode.add_argument("--snapshot", action="store_true",
                      help=f"keep the recorded versions under {SNAPSHOT.relative_to(ROOT)}/ before a run overwrites them")
    args = ap.parse_args()

    if args.snapshot:
        n = snapshot_recorded()
        print(f"{SNAPSHOT.relative_to(ROOT)}: {n} recorded outputs kept" if n else "recorded outputs already kept")
        return 0
    now = current()
    if args.write:
        MANIFEST.write_text("".join(f"{d}  {p}\n" for p, d in sorted(now.items())), encoding="utf-8", newline="\n")
        print(f"wrote {MANIFEST.relative_to(ROOT)}: {len(now)} files")
        return 0

    ref = read_manifest()
    changed = sorted(p for p in ref if p in now and now[p] != ref[p])
    missing = sorted(p for p in ref if p not in now)
    elsewhere = [p for p in missing if in_bundle(p) and not bundle_unpacked()]  # the fits, in the data bundle
    missing = [p for p in missing if p not in elsewhere]
    new = sorted(p for p in now if p not in ref)
    for label, paths in (("CHANGED", changed), ("MISSING", missing), ("NEW", new)):
        for p in paths:
            print(f"{label:8s} {p}")
    ok = not (changed or missing or new)
    print(f"{len(ref)} recorded, {len(now)} present: "
          + ("all identical" if ok else f"{len(changed)} changed, {len(missing)} missing, {len(new)} new")
          + (f" ({len(elsewhere)} model fits are in the data bundle, not unpacked)" if elsewhere else ""))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
