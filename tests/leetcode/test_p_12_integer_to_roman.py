import pytest

from leetcode.p_12_integer_to_roman import int_to_roman, int_to_roman_2


@pytest.mark.parametrize("input_value, expected", [(3749, "MMMDCCXLIX"), (58, "LVIII"), (1994, "MCMXCIV")])
def test_int_to_roman(input_value, expected):
    assert int_to_roman(input_value) == expected


@pytest.mark.parametrize("input_value, expected", [(3749, "MMMDCCXLIX"), (58, "LVIII"), (1994, "MCMXCIV")])
def test_int_to_roman_2(input_value, expected):
    assert int_to_roman_2(input_value) == expected
