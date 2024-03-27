import pytest
from leetcode.p_713_subarray_product_less_than_k import (
    num_subarray_product_less_than_k,
    num_subarray_product_less_than_k__double_loop,
)


@pytest.mark.parametrize(
    "input_arr, k, expected", [([1], 0, 0), ([1], 1, 0), ([1, 2, 3], 0, 0), ([10, 5, 2, 6], 100, 8)]
)
def test_subarray_product_less_than_k(input_arr, k, expected):
    assert num_subarray_product_less_than_k(input_arr, k) == expected


@pytest.mark.parametrize(
    "input_arr, k, expected", [([1], 0, 0), ([1], 1, 0), ([1, 2, 3], 0, 0), ([10, 5, 2, 6], 100, 8)]
)
def test_num_subarray_product_less_than_k__double_loop(input_arr, k, expected):
    assert num_subarray_product_less_than_k__double_loop(input_arr, k) == expected
