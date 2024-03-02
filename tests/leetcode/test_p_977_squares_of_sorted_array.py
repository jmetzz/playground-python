import pytest
from leetcode.p_977_squares_of_sorted_array import (
    sorted_squares_expanding_window,
    sorted_squares_naive,
    sorted_squares_shrinking_window,
)

INPUTS_AND_EXPECTATION = [
    ([-5, -3, -2, -1], [1, 4, 9, 25]),
    ([-5, -3, -2, -1, 0], [0, 1, 4, 9, 25]),
    ([-6, -5, -3, -2, -1], [1, 4, 9, 25, 36]),
    ([-4, -1, 0, 3, 10], [0, 1, 9, 16, 100]),
    ([-7, -3, 2, 3, 11], [4, 9, 9, 49, 121]),
]


@pytest.mark.parametrize("input_arr, expected", INPUTS_AND_EXPECTATION)
def test_sorted_squares_naive(input_arr, expected):
    assert sorted_squares_naive(input_arr) == expected


@pytest.mark.parametrize("input_arr, expected", INPUTS_AND_EXPECTATION)
def test_sorted_squares_expanding_win(input_arr, expected):
    assert sorted_squares_expanding_window(input_arr) == expected


@pytest.mark.parametrize("input_arr, expected", INPUTS_AND_EXPECTATION)
def test_sorted_squares_shrinking_win(input_arr, expected):
    assert sorted_squares_shrinking_window(input_arr) == expected
