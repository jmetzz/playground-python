import pytest
from leetcode.p_57_insert_intervals import insert_interval, insert_interval_2, insert_interval_3

test_cases = [
    ([[4, 5], [6, 9]], [0, 1], [[0, 1], [4, 5], [6, 9]]),
    ([[4, 5], [6, 9]], [12, 15], [[4, 5], [6, 9], [12, 15]]),
    ([[1, 3], [6, 9]], [2, 5], [[1, 5], [6, 9]]),
    ([[1, 2], [3, 5], [6, 7], [8, 10], [12, 16]], [4, 8], [[1, 2], [3, 10], [12, 16]]),
    ([[0, 2], [6, 9]], [3, 5], [[0, 2], [3, 5], [6, 9]]),
    ([[1, 3], [6, 9]], [5, 7], [[1, 3], [5, 9]]),
]


@pytest.mark.parametrize("func", [insert_interval, insert_interval_2, insert_interval_3])
@pytest.mark.parametrize("intervals, new_interval, expected", test_cases)
def test_insert_interval(func, intervals, new_interval, expected):
    assert func(intervals, new_interval) == expected
