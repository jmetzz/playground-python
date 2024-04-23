import pytest
from leetcode.p_2144_minimum_cost_of_buying_candies_with_discount import minimum_cost_loop, minimum_cost_stack


@pytest.mark.parametrize("func", [minimum_cost_stack, minimum_cost_loop])
@pytest.mark.parametrize("candies, expected", [([1, 2, 3], 5), ([6, 5, 7, 9, 2, 2], 23), ([5, 5], 10)])
def test_minimum_cost_to_buy_all_candies(func, candies, expected):
    assert func(candies[:]) == expected
