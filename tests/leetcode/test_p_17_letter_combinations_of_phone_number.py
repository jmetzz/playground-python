import pytest
from leetcode.p_17_letter_combinations_of_phone_number import (
    letter_combinations_backtracking,
    letter_combinations_iterative,
)


@pytest.mark.parametrize(
    "digits, expected",
    [
        ("23", ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"]),
        ("", []),
        ("2", ["a", "b", "c"]),
    ],
)
def test_letter_combinations_iterative(digits, expected):
    assert letter_combinations_iterative(digits) == expected


@pytest.mark.parametrize(
    "digits, expected",
    [
        ("23", ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"]),
        ("", []),
        ("2", ["a", "b", "c"]),
    ],
)
def test_letter_combinations_backtracking(digits, expected):
    assert letter_combinations_backtracking(digits) == expected
