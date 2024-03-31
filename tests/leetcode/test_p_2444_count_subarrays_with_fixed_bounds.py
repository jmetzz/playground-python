import pytest
from leetcode.p_2444_count_subarrays_with_fixed_bounds import count_subarrays


@pytest.mark.parametrize(
    "input_arr, min_k, max_k, expected",
    [
        ([1, 1, 1, 1], 1, 1, 10),
        ([1, 3, 5, 2, 7, 5], 1, 5, 2),
        ([1, 2, 5, 2, 1, 0, 7, 2, 5], 1, 5, 5),
    ],
)
def test_count_subarrays(input_arr, min_k, max_k, expected):
    assert count_subarrays(input_arr, min_k, max_k) == expected
