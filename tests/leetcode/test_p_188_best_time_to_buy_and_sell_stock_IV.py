import pytest

from leetcode.p_188_best_time_to_buy_and_sell_stock_IV import (
    list_transaction_maximizing_profit,
    max_profit_with_k_transactions,
)


@pytest.mark.parametrize(
    "prices, k,  expected",
    [
        ([2, 4, 1], 2, 2),
        ([3, 2, 6, 5, 0, 3], 2, 7),
        ([7, 6, 4, 3, 1], 2, 0),
        ([3, 3, 5, 0, 0, 3, 1, 4], 2, 6),
        ([1, 2, 3, 4, 5], 2, 4),
        ([2, 1, 2, 0, 1], 2, 2),
        ([1, 2, 4, 2, 5, 7, 2, 4, 9, 0], 2, 13),
        ([7, 1, 5, 2, 4, 8, 3, 10, 11, 12, 15, 14, 4, 2, 0], 2, 19),
        ([7, 1, 5, 2, 4, 8, 3, 10, 11, 12, 15, 14, 4, 2, 0], 1, 14),
    ],
)
def test_function_name(prices, k, expected):
    assert max_profit_with_k_transactions(prices, k) == expected


@pytest.mark.parametrize(
    "prices, k, expected_transactions",
    [
        # Test case format: (prices, k, expected_transactions)
        ([], 1, []),  # Empty price list
        ([5, 5, 5, 5], 2, []),  # No profit possible
        ([5, 5, 5, 5], 0, []),  # No transaction allowed
        ([5, 4, 3, 2, 1], 2, []),  # Monotonically decreasing prices
        ([2, 4, 1], 2, [(0, 1, 2)]),
        ([3, 3, 5, 0, 0, 3, 1, 4], 2, [(4, 5, 3), (6, 7, 3)]),
        # matching transactions available
        ([1, 2, 3, 4, 5], 2, [(0, 3, 3), (3, 4, 1)]),  # Monotonically increasing prices
        ([2, 1, 2, 0, 1], 2, [(1, 2, 1), (3, 4, 1)]),  # non-overlapping transactions
        ([3, 2, 6, 5, 0, 3], 2, [(1, 2, 4), (4, 5, 3)]),  # non-overlapping transactions
        ([1, 5, 2, 8, 3, 10], 3, [(0, 1, 4), (2, 3, 6), (4, 5, 7)]),  # non-overlapping transactions
        ([1, 2, 4, 2, 5, 7, 2, 4, 9, 0], 2, [(0, 5, 6), (6, 8, 7)]),  # overlapping transactions
        # more transactions avalable than requested
        (
            [7, 1, 5, 2, 4, 8, 3, 10, 11, 12, 15, 14, 4, 2, 0],  # Multiple profitable transactions
            1,
            [(1, 10, 14)],
        ),
        (
            [7, 1, 5, 2, 4, 8, 3, 10, 11, 12, 15, 14, 4, 2, 0],  # Multiple profitable transactions
            2,
            [(1, 5, 7), (6, 10, 12)],
        ),
        (
            [7, 1, 5, 2, 4, 8, 3, 10, 11, 12, 15, 14, 4, 2, 0],  # Multiple profitable transactions
            3,
            [(1, 2, 4), (3, 5, 6), (6, 10, 12)],
        ),
        (
            [7, 1, 5, 2, 4, 8, 3, 10, 11, 12, 15, 14, 4, 2, 0, 1, 5, 7, 4, 0],  # infinit transactions
            9999,
            [
                (1, 2, 4),
                (3, 4, 2),
                (4, 5, 4),
                (6, 7, 7),
                (7, 8, 1),
                (8, 9, 1),
                (9, 10, 3),
                (14, 15, 1),
                (15, 16, 4),
                (16, 17, 2),
            ],
        ),
    ],
)
def test_list_transaction(prices, k, expected_transactions):
    assert list_transaction_maximizing_profit(prices, k) == expected_transactions
