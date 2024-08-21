import pytest

from leetcode.p_80_remove_duplicates_from_sorted_array_II import remove_duplicates, remove_duplicates_2


@pytest.mark.parametrize("func", [remove_duplicates, remove_duplicates_2])
@pytest.mark.parametrize(
    "input_array, expected_arr, expected_size",
    [
        ([], [], 0),
        ([1], [1], 1),
        ([1, 2], [1, 2], 2),
        ([1, 1], [1, 1], 2),
        ([1, 2, 2], [1, 2, 2], 3),
        ([1, 1, 2], [1, 1, 2], 3),
        ([1, 1, 1, 2], [1, 1, 2, 2], 3),
        ([0, 0, 1, 1, 1, 1, 2, 3, 3], [0, 0, 1, 1, 2, 3, 3, 3, 3], 7),
    ],
)
def test_remove_duplicates(func, input_array, expected_arr, expected_size):
    arr_cp = input_array[:]
    actual_k = func(arr_cp)
    assert arr_cp == expected_arr
    assert actual_k == expected_size
