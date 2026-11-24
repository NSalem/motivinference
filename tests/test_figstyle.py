"""Unit tests for the figure house style (``motinf.exploratory.figlib``): units, labels and significance marks."""

from __future__ import annotations


import matplotlib

matplotlib.use("Agg")

from motinf.exploratory.figlib import dist, figstyle as st


def test_mm_to_inches() -> None:
    assert st.mm(25.4) == 1.0
    assert st.mm(st.DOUBLE, 50.8) == (st.DOUBLE / 25.4, 2.0)


def test_page_geometry_is_consistent() -> None:
    assert st.SINGLE < st.MID < st.DOUBLE
    assert st.SIZE_MIN <= st.SIZE_TEXT < st.SIZE_LABEL <= st.SIZE_MAX < st.SIZE_LETTER


def test_page_scale_maps_the_layout_to_the_journal_width() -> None:
    assert st.LAYOUT_DOUBLE * st.page_scale(st.LAYOUT_DOUBLE) == st.DOUBLE
    assert st.page_scale(st.SINGLE) == 1.0  # single-column figures are laid out at their printed size
    assert st.DOUBLE <= 180.0  # NHB: figures at most 180 mm wide


def test_panel_letters_are_lowercase() -> None:
    assert st.letter(1) == "b"


def test_fmt_num_true_minus_and_no_negative_zero() -> None:
    assert st.fmt_num(-0.001) == "0.00"
    assert st.fmt_num(-0.25) == "−0.25"
    assert st.fmt_num(0.256, nd=1) == "0.3"


def test_sym_uses_sensory_subscript() -> None:
    assert st.sym("sigma", "inf") == r"$\sigma_{\mathrm{inf}}$"
    assert st.sym("delta", "sen") == r"$\delta_{\mathrm{ang}}$"


def test_p_stars_thresholds() -> None:
    assert dist.p_stars(0.2) == ""
    assert dist.p_stars(0.05) == ""
    assert dist.p_stars(0.049) == "*"
    assert dist.p_stars(0.0099) == "**"
    assert dist.p_stars(0.0009) == "***"
    assert dist.p_stars(float("nan")) == ""


def test_text_is_black_and_incentive_colours_are_the_papers() -> None:
    assert st.TEXT == "#000000"
    assert st.INC_COLORS == {st.LOSS: "#e41a1c", st.NEUTRAL: "#8c8c8c", st.GAIN: "#4daf4a"}
