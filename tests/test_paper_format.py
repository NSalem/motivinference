"""The number-formatting rules of the paper (``motinf.exploratory.paper_format``)."""

from __future__ import annotations

import pytest
from scipy.stats import gamma, norm

from motinf.exploratory.paper_format import apply, bf, bf_number, normalize, num, p_value, sci, signed, stars


def test_num_true_minus_and_no_negative_zero() -> None:
    assert num(-0.294, 2) == "−0.29"
    assert num(-0.001, 2) == "0.00"
    assert signed(8.2) == "+8" and signed(-5.4) == "−5" and signed(0.2) == "0"


@pytest.mark.parametrize("p, s", [(0.00099, "< 0.001"), (0.001, "0.001"), (0.0734, "0.073"), (0.9996, "1.000"),
                                  (1.0, "1.000")])
def test_p_value(p: float, s: str) -> None:
    assert p_value(p) == s


def test_stars_thresholds() -> None:
    assert [stars(p) for p in (0.0009, 0.009, 0.049, 0.05)] == ["***", "**", "*", ""]


@pytest.mark.parametrize("b, s", [(2.14, "2.1"), (9.96, "10.0"), (10.4, "10"), (536.2, "536"),
                                  (999.4, "999"), (999.6, "1.0×10³"), (3.94e3, "3.9×10³"), (9.96e3, "1.0×10⁴"),
                                  (6.5e34, "6.5×10³⁴")])
def test_bf_number(b: float, s: str) -> None:
    assert bf_number(b) == s


def test_bf_picks_the_direction_above_one() -> None:
    assert bf(141.2) == "BF₁₀ = 141"
    assert bf(1 / 4.7) == "BF₀₁ = 4.7"


def test_sci_negative_exponent() -> None:
    assert sci(9.943e-5, 2) == "9.94×10⁻⁵"
    assert sci(6.0209e-23, 2) == "6.02×10⁻²³"


def test_rules() -> None:
    assert apply("num_floor(2)", [0.004]) == "< 0.01"
    assert apply("num_floor(2)", [0.0779]) == "0.08"
    assert apply("p_min_floor(2)", [0.254, 0.31]) == "0.25"
    assert apply("ci(2)", [-0.433, -0.164]) == "[−0.43, −0.16]"
    assert apply("num_stars(2)", [0.334, 0.0004]) == "0.33***"
    assert apply("df", [2.0, 166.0]) == "2, 166"
    assert apply("int_comma", [72576.0]) == "72,576"
    assert apply("word", [4.0]) == "four"
    assert apply("range(0, 100)", [0.921, 0.990]) == "92–99"
    assert apply("range_abs(2)", [-0.32, -0.29, -0.40]) == "−0.29 to −0.40"
    assert apply("bf01_range", [1 / 6.9, 1 / 13.2, 1 / 9.8]) == "6.9–13"
    assert apply("gt(0.99)", [0.9999]) == "0.99"
    assert apply("gt(0.99)", [0.98]).startswith("not")
    assert apply("changed_p(word)", [0.2895, 0.0689, 0.0002, 0.2964, 0.0725, 0.0002]) == "two"
    assert apply("bounds", [1e-6, 5]) == "[10⁻⁶, 5]"
    assert apply("gamma", [gamma(a=2, scale=0.5)]) == "Gamma(2, 0.5)"
    assert apply("normal", [norm(loc=0, scale=0.5)]) == "N(0, 0.5)"
    with pytest.raises(KeyError):
        apply("no_such_rule", [1.0])


def test_normalize_ignores_typography_only() -> None:
    assert normalize("[-3, 3]") == normalize("[−3, 3]")
    assert normalize("N(0,0.5)") == normalize("N(0, 0.5)")
    assert normalize("92–99") != normalize("92−99")  # a range dash is not a minus sign
    assert normalize("0.33") != normalize("0.34")
