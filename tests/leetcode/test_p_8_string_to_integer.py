import pytest
from leetcode.p_8_string_to_integer import my_atoi


@pytest.mark.parametrize(
    "input_string, expected",
    [("42", 42), ("   -42", -42), ("4193 with words", 4193), ("", 0), ("+-12", 0), ("00000-42a1234", 0)],
)
def test_my_atoi(input_string, expected):
    assert my_atoi(input_string) == expected
