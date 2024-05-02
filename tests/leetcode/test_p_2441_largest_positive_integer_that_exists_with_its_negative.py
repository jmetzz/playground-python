import pytest
from leetcode.p_2441_largest_positive_integer_that_exists_with_its_negative import (
    find_max_k__set_intersection,
    find_max_k__set_removal,
)


@pytest.mark.parametrize("func", [find_max_k__set_intersection, find_max_k__set_removal])
@pytest.mark.parametrize(
    "input_arr, expected",
    [
        ([-1, 2, -3, 3], 3),
        ([-1, 10, 6, 7, -7, 1], 7),
        ([-10, 8, 6, 7, -2, -3], -1),
    ],
)
def test_find_max_k(func, input_arr, expected):
    assert func(input_arr) == expected
