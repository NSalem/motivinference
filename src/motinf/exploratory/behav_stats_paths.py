"""Output paths of ``run_behavior_stats.py``."""

from __future__ import annotations

from pathlib import Path


def behav_extra_dir(root: Path, exp_name: str) -> Path:
    """Root directory for ``run_behavior_stats.py`` outputs."""
    return root / "results" / "exploratory" / exp_name.strip() / "behavior" / "stats" / "behav_extra"
