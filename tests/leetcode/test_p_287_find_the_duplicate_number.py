import pytest
from leetcode.p_287_find_the_duplicate_number import (
    find_duplicate_bitwise,
    find_duplicate_bool_array,
    find_duplicate_brute_force,
    find_duplicate_builtin_counter,
    find_duplicate_dict_counter,
    find_duplicate_fast_slow_pointers,
    find_duplicate_hashset,
    find_duplicate_sorting,
    find_duplicate_subtract,
)


@pytest.mark.parametrize(
    "input_arr, expected",
    [
        ([1, 1], 1),
        ([2, 2, 1], 2),
        ([2, 1, 1], 1),
        ([1, 3, 4, 2, 2], 2),
        ([3, 3, 3, 3, 3], 3),
        ([3, 1, 3, 4, 3], 3),
        ([4, 3, 1, 4, 2], 4),
        ([1, 2, 3, 4, 4], 4),
        ([1, 2, 3, 4, 5, 2], 2),
        ([5, 4, 3, 2, 1, 1], 1),
        ([2, 6, 4, 1, 3, 1, 5], 1),
        ([10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1], 1),
    ],
)
@pytest.mark.parametrize(
    "func",
    [
        find_duplicate_brute_force,
        find_duplicate_sorting,
        find_duplicate_hashset,
        find_duplicate_bool_array,
        find_duplicate_builtin_counter,
        find_duplicate_dict_counter,
        find_duplicate_subtract,
        find_duplicate_fast_slow_pointers,
        find_duplicate_bitwise,
    ],
)
def test_find_duplicate(func, input_arr, expected):
    assert func(input_arr) == expected
