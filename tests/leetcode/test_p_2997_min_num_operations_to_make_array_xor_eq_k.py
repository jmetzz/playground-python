import pytest

from leetcode.p_2997_min_num_operations_to_make_array_xor_eq_k import min_operations, min_perations_iterative


@pytest.mark.parametrize("func", [min_operations, min_perations_iterative])
@pytest.mark.parametrize("input_arr, k, expected", [([2, 1, 3, 4], 1, 2), ([2, 0, 2, 0], 0, 0)])
def test_min_operations(func, input_arr, k, expected):
    assert func(input_arr, k) == expected
