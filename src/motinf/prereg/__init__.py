"""Preregistered analysis namespace."""

from .aggregation import aggregate_data_prereg
from .behavior import run_behavior_main_prereg
from .cleaning import QCConfigPrereg, run_cleaning_prereg

__all__ = [
    "QCConfigPrereg",
    "aggregate_data_prereg",
    "run_cleaning_prereg",
    "run_behavior_main_prereg",
]
