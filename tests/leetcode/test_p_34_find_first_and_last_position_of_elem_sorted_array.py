import pytest

from leetcode.p_34_find_first_and_last_position_of_elem_sorted_array import (
    search_range_bin_search,
    search_range_bisect,
    search_range_find,
)

test_cases = [
    ([], 0, [-1, -1]),
    ([1], 0, [-1, -1]),
    ([0], 0, [0, 0]),
    ([0, 1], 0, [0, 0]),
    ([0, 0, 1], 0, [0, 1]),
    ([0, 0, 1], 1, [2, 2]),
    ([0, 0, 1, 1], 1, [2, 3]),
    ([5, 7, 7, 8, 8, 10], 8, [3, 4]),
    ([5, 7, 7, 8, 8, 10], 6, [-1, -1]),
    ([5, 7, 7, 8, 8, 8, 8, 9, 10], 9, [7, 7]),
    ([5, 7, 7, 8, 8, 8, 8, 9, 9, 10], 9, [7, 8]),
]


@pytest.mark.parametrize("input_arr, input_target, expected", test_cases)
def test_search_range_bisect(input_arr, input_target, expected):
    assert search_range_bisect(input_arr, input_target) == expected


@pytest.mark.parametrize("input_arr, input_target, expected", test_cases)
def test_search_range_find(input_arr, input_target, expected):
    assert search_range_find(input_arr, input_target) == expected


@pytest.mark.parametrize("input_arr, input_target, expected", test_cases)
def test_search_range_bin_search(input_arr, input_target, expected):
    assert search_range_bin_search(input_arr, input_target) == expected
