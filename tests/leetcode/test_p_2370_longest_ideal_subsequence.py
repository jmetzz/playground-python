import pytest

from leetcode.p_2370_longest_ideal_subsequence import (
    longest_ideal_string_dp_recursive,
    longest_ideal_string_dp_space_optimized,
    longest_ideal_string_window,
)


@pytest.mark.parametrize(
    "func", [longest_ideal_string_dp_recursive, longest_ideal_string_dp_space_optimized, longest_ideal_string_window]
)
@pytest.mark.parametrize("word, k, expected", [("acfgbd", 2, 4), ("abcd", 3, 4)])
def test_longest_ideal_string(func, word, k, expected):
    assert func(word, k) == expected
