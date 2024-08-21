import pytest

from leetcode.p_121_best_time_to_buy_and_sell_stock import max_profit_1, max_profit_2, max_profit_3


@pytest.mark.parametrize("func", [max_profit_1, max_profit_2, max_profit_3])
@pytest.mark.parametrize("prices, expected", [([7, 1, 5, 3, 6, 4], 5), ([7, 6, 4, 3, 1], 0)])
def test_max_profit(func, prices, expected):
    assert func(prices) == expected
