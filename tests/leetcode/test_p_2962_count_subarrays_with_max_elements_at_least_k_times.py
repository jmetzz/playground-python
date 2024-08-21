import pytest

from leetcode.p_2962_count_subarrays_with_max_elements_at_least_k_times import count_subarrays


@pytest.mark.parametrize(
    "input_arr, k, expected", [([1, 3, 2, 3, 3], 2, 6), ([1, 4, 2, 1], 3, 0), ([1, 4, 2, 4, 3, 4], 2, 8)]
)
def test_count_subarrays_with_max_at_least_k_times(input_arr, k, expected):
    assert count_subarrays(input_arr, k) == expected
