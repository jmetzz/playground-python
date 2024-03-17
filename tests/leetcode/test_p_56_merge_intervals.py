import pytest
from leetcode.p_56_merge_intervals import merge_intervals, merge_intervals_2

test_cases = [
    ([[0, 1], [4, 5], [6, 9]], [[0, 1], [4, 5], [6, 9]]),  # no overlap
    ([[4, 5], [5, 9], [12, 15]], [[4, 9], [12, 15]]),  # 2 overlapping on the boundary
    ([[1, 5], [3, 9]], [[1, 9]]),  # 2 overlapping
    ([[1, 2], [3, 4], [5, 6], [8, 10], [12, 16]], [[1, 2], [3, 4], [5, 6], [8, 10], [12, 16]]),  # no overlap
    ([[1, 2], [3, 6], [5, 6], [6, 9], [8, 10], [12, 16]], [[1, 2], [3, 10], [12, 16]]),  #  4 overlapping
]


@pytest.mark.parametrize("func", [merge_intervals, merge_intervals_2])
@pytest.mark.parametrize("intervals, expected", test_cases)
def test_merge_intervals(func, intervals, expected):
    assert func(intervals) == expected
