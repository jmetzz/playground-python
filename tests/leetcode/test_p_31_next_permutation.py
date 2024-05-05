import pytest
from leetcode.p_31_next_permutation import next_permutation


@pytest.mark.parametrize(
    "input_arr, expected", [([1, 2, 3], [1, 3, 2]), ([3, 2, 1], [1, 2, 3]), ([1, 1, 5], [1, 5, 1]), ([1, 2], [2, 1])]
)
def test_next_permutation(input_arr, expected):
    # the function under tests perform in_place operation
    next_permutation(input_arr)
    assert input_arr == expected
