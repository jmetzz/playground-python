import pytest
from leetcode.p_2966_divide_array_into_arrays_with_max_diff import (
    DivideArraysWithMaxDiff,
)


@pytest.mark.parametrize(
    "test_input_arr, test_input_k, expected",
    [
        ([1, 3, 4, 8, 7, 9, 3, 5, 1], 2, [[3, 1, 1], [5, 4, 3], [9, 8, 7]]),
        ([1, 3, 3, 2, 7, 3], 3, []),
    ],
)
def test_divide_array_with_max_diff_backward(test_input_arr, test_input_k, expected):
    assert DivideArraysWithMaxDiff().solve_backward(test_input_arr, test_input_k) == expected


@pytest.mark.parametrize(
    "test_input_arr, test_input_k, expected",
    [
        ([1, 3, 4, 8, 7, 9, 3, 5, 1], 2, [[1, 1, 3], [3, 4, 5], [7, 8, 9]]),
        ([1, 3, 3, 2, 7, 3], 3, []),
    ],
)
def test_divide_array_with_max_diff_forward(test_input_arr, test_input_k, expected):
    assert DivideArraysWithMaxDiff().solve_forward(test_input_arr, test_input_k) == expected
