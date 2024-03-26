import pytest
from leetcode.p_41_first_missing_positive import (
    first_missing_positive_boolean_array,
    first_missing_positive_cycle_sort,
    first_missing_positive_filter,
    first_missing_positive_set,
    first_missing_positive_side_effect,
)

test_cases = [
    ([], 1),
    ([1], 2),
    ([1, 1], 2),
    ([2, 2], 1),
    ([1, 2, 3], 4),
    ([-1, -2, -3], 1),
    ([3, 4, -1, 1], 2),
    ([7, 8, 9, 11, 12], 1),
    ([0, 0, 2, 2, 3, 3], 1),
    ([2, 3, 4, 5, 6, 7, 8, 9, 1], 10),
]


@pytest.mark.parametrize(
    "func",
    [
        first_missing_positive_filter,
        first_missing_positive_set,
        first_missing_positive_boolean_array,
        first_missing_positive_side_effect,
        first_missing_positive_cycle_sort,
    ],
)
@pytest.mark.parametrize("input_arr, expected", test_cases)
def test_first_missing_positive(func, input_arr, expected):
    # pass a copy of the input_arr since some of the implementations
    # operate on side-effects
    assert func(input_arr[:]) == expected
