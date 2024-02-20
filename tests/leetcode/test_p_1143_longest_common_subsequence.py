import pytest
from leetcode.p_1143_longest_common_subsequence import LongestCommonSubsequence


@pytest.mark.parametrize(
    "test_input_1, test_input_2, expected",
    [
        ("abcde", "ace", 3),
        ("abc", "abc", 3),
        ("abc", "def", 0),
    ],
)
def test_longest_commmon_subsequence(test_input_1, test_input_2, expected):
    assert LongestCommonSubsequence().solve(test_input_1, test_input_2) == expected
