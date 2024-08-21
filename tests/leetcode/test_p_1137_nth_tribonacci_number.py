import pytest

from leetcode.p_1137_nth_tribonacci_number import (
    tribonacci_dp_space_order_1_list,
    tribonacci_dp_space_order_1_vars,
    tribonacci_dp_space_order_n,
    tribonacci_recursive_memoization,
)


@pytest.mark.parametrize(
    "func",
    [
        tribonacci_dp_space_order_1_list,
        tribonacci_dp_space_order_1_vars,
        tribonacci_dp_space_order_n,
        tribonacci_recursive_memoization,
    ],
)
@pytest.mark.parametrize("input_data, expected", [(4, 4), (25, 1389537)])
def test_tribonacci(func, input_data, expected):
    assert func(input_data) == expected
