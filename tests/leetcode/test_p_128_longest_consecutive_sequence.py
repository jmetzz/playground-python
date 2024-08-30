import pytest

from leetcode.p_128_longest_consecutive_sequence import (
    longest_consecutive,
    longest_consecutive_naive,
    longest_consecutive_perf_improvement,
    longest_consecutive_union_find,
)


@pytest.mark.parametrize(
    "func",
    [
        longest_consecutive,
        longest_consecutive_naive,
        longest_consecutive_perf_improvement,
        longest_consecutive_union_find,
    ],
)
@pytest.mark.parametrize(
    "input_data, expected",
    [
        ([], 0),
        ([100, 4, 200, 1, 3, 2], 4),
        ([0, 3, 7, 2, 5, 8, 4, 6, 0, 1], 9),
    ],
)
def test_function_name(func, input_data, expected):
    assert func(input_data) == expected
