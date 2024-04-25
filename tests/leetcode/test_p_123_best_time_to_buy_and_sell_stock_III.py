import pytest
from leetcode.p_123_best_time_to_buy_and_sell_stock_III import (
    max_profit_heap,
    max_profit_list_transactions,
    max_profit_memoization,
    max_profit_recursive,
    max_profit_space_optmized,
    max_profit_tabulation,
    max_profit_two_segments,
)


@pytest.mark.parametrize(
    "func",
    [
        max_profit_two_segments,
        max_profit_recursive,
        max_profit_memoization,
        max_profit_tabulation,
        max_profit_space_optmized,
    ],
)
@pytest.mark.parametrize(
    "input_data, expected",
    [
        ([7, 6, 4, 3, 1], 0),
        ([3, 3, 5, 0, 0, 3, 1, 4], 6),
        ([1, 2, 3, 4, 5], 4),
        ([2, 1, 2, 0, 1], 2),
        ([1, 2, 4, 2, 5, 7, 2, 4, 9, 0], 13),
    ],
)
def test_best_time_to_buy_and_sell_stock_III(func, input_data, expected):
    assert func(input_data) == expected


@pytest.mark.parametrize(
    "func",
    [
        max_profit_heap,
        max_profit_two_segments,
        max_profit_recursive,
        max_profit_memoization,
        max_profit_tabulation,
        max_profit_space_optmized,
    ],
)
@pytest.mark.parametrize(
    "input_data, expected",
    [
        ([7, 6, 4, 3, 1], 0),
        ([3, 3, 5, 0, 0, 3, 1, 4], 6),
        ([1, 2, 3, 4, 5], 4),
        ([2, 1, 2, 0, 1], 2),
    ],
)
def test_best_time_to_buy_and_sell_stock_III_non_fluctuating_intervals(func, input_data, expected):
    assert func(input_data) == expected


@pytest.mark.parametrize(
    "input_data, expected",
    [
        ([7, 6, 4, 3, 1], []),
        ([3, 3, 5, 0, 0, 3, 1, 4], [(3, 3, 5), (3, 6, 7)]),
        ([1, 2, 3, 4, 5], [(4, 0, 4)]),
        ([2, 1, 2, 0, 1], [(1, 1, 2), (1, 3, 4)]),
    ],
)
def test_list_transactions_maximizing_profit(input_data, expected):
    assert max_profit_list_transactions(input_data) == expected
