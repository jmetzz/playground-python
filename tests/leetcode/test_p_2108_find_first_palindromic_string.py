import pytest
from leetcode.p_2108_find_first_palindromic_string import FirstPalindrome


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (["abc", "car", "ada", "racecar", "cool"], "ada"),
        (["notapalindrome", "racecar"], "racecar"),
        (["def", "ghi"], ""),
        (["xngla", "e", "itrn", "y", "s", "pfp", "rfd"], "e"),
    ],
)
def test_find_first_palindromic_str(test_input, expected):
    assert FirstPalindrome().solve(test_input) == expected
