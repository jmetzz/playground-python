# print(final_prices([8, 4, 6, 2, 3]))  # 4,2,4,2,3
# print(final_prices([1, 2, 3, 4, 5]))  # [1,2,3,4,5]
# print(final_prices([10, 1, 1, 6]))  # [9,0,1,6]

import pytest

from leetcode.p_1475_final_prices_with_special_discount_in_shop import final_prices, final_prices_greedy


@pytest.mark.parametrize("func", [final_prices, final_prices_greedy])
@pytest.mark.parametrize(
    "input_prices_arr, expected",
    [([8, 4, 6, 2, 3], [4, 2, 4, 2, 3]), ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]), ([10, 1, 1, 6], [9, 0, 1, 6])],
)
def test_final_prices(func, input_prices_arr, expected):
    assert func(input_prices_arr) == expected
