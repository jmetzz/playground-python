import pytest

from leetcode.p_88_merge_sorted_array import merge_inplace, merge_inplace_2

test_cases = [
    ([0], 0, [1], 1, [1]),
    ([1], 1, [], 0, [1]),
    ([1, 2, 3, 0, 0, 0], 3, [2, 5, 6], 3, [1, 2, 2, 3, 5, 6]),
    ([2, 5, 6, 0, 0, 0], 3, [1, 2, 3], 3, [1, 2, 2, 3, 5, 6]),
]


@pytest.mark.parametrize("input_arr1, m, input_arr2, n, expected", test_cases)
def test_merge_inplace(input_arr1, m, input_arr2, n, expected):
    arr1 = input_arr1[:]  # to avoid side-effects from inplace changes
    merge_inplace(arr1, m, input_arr2, n)
    assert arr1 == expected


@pytest.mark.parametrize("input_arr1, m, input_arr2, n, expected", test_cases)
def test_merge_inplace_2(input_arr1, m, input_arr2, n, expected):
    arr1 = input_arr1[:]  # to avoid side-effects from inplace changes
    merge_inplace_2(arr1, m, input_arr2, n)
    assert arr1 == expected
