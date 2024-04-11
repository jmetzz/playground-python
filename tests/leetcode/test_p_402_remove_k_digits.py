import pytest
from leetcode.p_402_remove_k_digits import remove_k_digits


@pytest.mark.parametrize(
    "input_str, k, expected",
    [
        ("", 2, "0"),
        ("1", 2, "0"),
        ("10", 2, "0"),
        ("10200", 1, "200"),
        ("1432219", 1, "132219"),
        ("1432219", 2, "12219"),
        ("1432219", 3, "1219"),
        ("1432219", 4, "119"),
        ("1432219", 5, "11"),
        ("1432219", 6, "1"),
        ("1432219", 7, "0"),
        ("1432219", 100, "0"),
    ],
)
def test_remove_k_digits(input_str, k, expected):
    assert remove_k_digits(input_str, k) == expected
