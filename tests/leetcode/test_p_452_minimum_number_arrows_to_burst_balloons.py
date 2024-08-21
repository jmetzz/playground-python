import pytest

from leetcode.p_452_minimum_number_arrows_to_burst_balloons import find_min_arrow_shots, find_min_arrow_shots_2

test_cases = [
    ([[1, 2]], 1),
    ([[1, 2], [3, 4]], 2),
    ([[1, 2], [2, 4], [2, 6], [1, 8]], 1),
    ([[10, 16], [2, 8], [1, 6], [7, 12]], 2),
    ([[1, 2], [2, 3], [3, 4], [4, 5]], 2),
    ([[1, 2], [3, 4], [5, 6], [7, 8]], 4),
]


@pytest.mark.parametrize("func", [find_min_arrow_shots, find_min_arrow_shots_2])
@pytest.mark.parametrize("intervals, expected", test_cases)
def test_find_min_arrow(func, intervals, expected):
    assert func(intervals) == expected
