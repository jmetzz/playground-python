import pytest
from leetcode.p_992_subarrays_with_k_diff_integers import subarrays_with_k_distinct


@pytest.mark.parametrize("input_arr, k, expected", [([1, 2, 1, 2, 3], 2, 7), ([1, 2, 1, 3, 4], 3, 3)])
def test_subarrays_with_k_distinct(input_arr, k, expected):
    assert subarrays_with_k_distinct(input_arr, k) == expected
