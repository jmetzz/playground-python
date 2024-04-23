import pytest
from leetcode.p_2706_buy_two_chocolates import buy_choco


@pytest.mark.parametrize(
    "prices, money, expected",
    [
        ([], 20, 20),  # No chocolates to buy, return the money
        ([2, 2, 2, 2], 0, 0),  # no money, cannot buy chocos
        ([1, 2, 3, 20], 7, 4),  # buy 1 and 2, balance 4
        ([20, 10, 5, 1], 5, 5),  # cannot buy 2 chocos, return the money
        ([5, 3, 9], 10, 2),  # buy 3 and 5, balance 2
        ([10, 2, 6, 2], 10, 6),  # buy 2 and 2, balance 6
        ([3, 3, 3, 3], 12, 6),  # buy 2 chocos at 3, balance 6
        ([15, 20, 30], 5, 5),  # cannot buy any choco, return the money
    ],
)
def test_buy_choco(prices, money, expected):
    assert buy_choco(prices, money) == expected
