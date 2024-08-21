import pytest

from leetcode.p_238_product_of_array_except_self import product_except_self


@pytest.mark.parametrize(
    "input_arr, expected",
    [
        ([1, 2, 3, 4], [24, 12, 8, 6]),
        ([-1, 1, 0, -3, 3], [0, 0, 9, 0, 0]),
        ([2, 3], [3, 2]),
        ([1, 2, 3, 4, 5], [120, 60, 40, 30, 24]),
        ([-2, -1, 0, 1, 2], [0, 0, 4, 0, 0]),
    ],
)
def test_product_except_self(input_arr, expected):
    assert product_except_self(input_arr) == expected
