import pytest

from leetcode.p_31_next_permutation import next_permutation, next_permutation_review


@pytest.mark.parametrize(
    "input_arr, expected", [([1, 2, 3], [1, 3, 2]), ([3, 2, 1], [1, 2, 3]), ([1, 1, 5], [1, 5, 1]), ([1, 2], [2, 1])]
)
def test_next_permutation(input_arr, expected):
    # the function under tests perform in_place operation
    next_permutation(input_arr)
    assert input_arr == expected


@pytest.mark.parametrize(
    "input_arr, expected",
    [
        ([1, 2, 3], [1, 3, 2]),
        ([3, 2, 1], [1, 2, 3]),
        ([1, 1, 5], [1, 5, 1]),
        ([1, 2], [2, 1]),
        ([1, 5, 8, 4, 7, 6, 5, 3, 1], [1, 5, 8, 5, 1, 3, 4, 6, 7]),
    ],
)
def test_next_permutation_review(input_arr, expected):
    # the function under tests perform in_place operation
    next_permutation_review(input_arr)
    assert input_arr == expected
