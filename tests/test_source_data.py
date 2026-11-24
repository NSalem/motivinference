"""The figure source-data writer: long form, change from neutral, index, no stale files."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from motinf.exploratory.figlib.source_data import SourceData


def test_levels_long_form_and_change(tmp_path) -> None:
    wide = pd.DataFrame({-1.0: [2.0, 4.0], 0.0: [1.0, 2.0], 1.0: [0.5, 8.0]}, index=pd.Index(["S1", "S2"]))
    sd = SourceData("figX", tmp_path)
    sd.add_levels("a", wide, "v", "values")
    sd.add_levels("b", wide, "v", "values", change="logdiff")
    sd.note("c", "schematic")
    out = sd.write()
    a = pd.read_csv(out / "figXa.csv")
    assert list(a.columns) == ["participant", "incentive", "v", "v_change_from_neutral"]
    assert len(a) == 6
    s2 = a[a["participant"] == "S2"].set_index("incentive")
    assert s2.loc[1.0, "v_change_from_neutral"] == 6.0 and np.isnan(s2.loc[0.0, "v_change_from_neutral"])
    b = pd.read_csv(out / "figXb.csv").set_index(["participant", "incentive"])
    assert b.loc[("S1", -1.0), "log_v_change_from_neutral"] == pytest.approx(np.log(2.0))
    index = pd.read_csv(out / "index.csv", keep_default_na=False)
    assert list(index["file"]) == ["figXa.csv", "figXb.csv", ""]


def test_rewrite_removes_stale_files_and_rejects_duplicates(tmp_path) -> None:
    sd = SourceData("figY", tmp_path)
    sd.add("a", {"x": [1]}, "one")
    out = sd.write()
    (out / "figYz.csv").write_text("old\n")
    sd.write()
    assert sorted(p.name for p in out.glob("*.csv")) == ["figYa.csv", "index.csv"]
    with pytest.raises(ValueError):
        sd.add("a", {"x": [2]}, "again")
