import pytest
from leetcode.p_122_best_time_to_buy_and_sell_stock_II import max_profit


@pytest.mark.parametrize("input_data, expected", [([7, 1, 5, 3, 6, 4], 7), ([1, 2, 3, 4, 5], 4), ([7, 6, 4, 3, 1], 0)])
def test_function_name(input_data, expected):
    assert max_profit(input_data) == expected
