"""
https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/description/

122. Best Time to Buy and Sell Stock II
Medium

You are given an integer array prices where prices[i] is the price of a given stock on the ith day.

On each day, you may decide to buy and/or sell the stock.
You can only hold at most one share of the stock at any time.
However, you can buy it then immediately sell it on the same day.

Find and return the maximum profit you can achieve.



Example 1:

Input: prices = [7,1,5,3,6,4]
Output: 7
Explanation: Buy on day 2 (price = 1) and sell on day 3 (price = 5), profit = 5-1 = 4.
Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = 6-3 = 3.
Total profit is 4 + 3 = 7.
Example 2:

Input: prices = [1,2,3,4,5]
Output: 4
Explanation: Buy on day 1 (price = 1) and sell on day 5 (price = 5), profit = 5-1 = 4.
Total profit is 4.
Example 3:

Input: prices = [7,6,4,3,1]
Output: 0
Explanation: There is no way to make a positive profit,
so we never buy the stock to achieve the maximum profit of 0.


Constraints:

1 <= prices.length <= 3 * 10**4
0 <= prices[i] <= 10**4
"""

from typing import List


def max_profit(prices: List[int]) -> int:
    """
        Calculate the maximum profit from a given list of stock prices,

        where each day allows for buying or selling the stock.
        The strategy is to buy on one day and sell on any subsequent day
        that offers a higher price, continuing this pattern throughout
        the list of prices.

        To better understand the intuition for the solution, let's plot some values
        [7, 1, 5, 3, 6, 4]

        Price
        7 |   x
        6 |                   x
        5 |           x
        4 |                    x
        3 |               x
        2 |
        1 |       x
           -------------------------
    Day:      1   2   3   4   5   6
                  B   S   B   S

        B: represents Buy
        S: represents Sell


        The idea is to accumulate profits when the price trend is moving upwards
        in the price list, simulating a buy on the previous day and a sell on
        the current day if the price has increased.
        For the example above, from day 1 to day 2 the trend is going down.
        Thus, we don't buy it. From day 2 to day 3, the trend is upwards and thus
        we buy it on day 2 because we know we can make profit on the next day.


        Parameters:
        - prices (List[int]): A list of integers representing the stock prices on consecutive days.

        Returns:
        - int: The maximum profit that can be achieved based on the given trading strategy.

        Example:
        >>> max_profit([7, 1, 5, 3, 6, 4])
        7
    """
    profit = 0
    p = prices[0]
    for idx in range(1, len(prices)):
        if prices[idx] > p:
            profit += prices[idx] - p
        p = prices[idx]
    return profit
