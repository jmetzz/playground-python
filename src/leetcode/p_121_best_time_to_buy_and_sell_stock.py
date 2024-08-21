"""https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/

121. Best Time to Buy and Sell Stock
Easy

You are given an array prices where prices[i] is the price of a given stock on the ith day.

You want to maximize your profit by choosing a single day to buy one stock and choosing a
different day in the future to sell that stock.

Return the maximum profit you can achieve from this transaction.
If you cannot achieve any profit, return 0.



Example 1:
Input: prices = [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.

Example 2:
Input: prices = [7,6,4,3,1]
Output: 0
Explanation: In this case, no transactions are done and the max profit = 0.


Constraints:

1 <= prices.length <= 10**5
0 <= prices[i] <= 10**4
"""

from typing import List


def max_profit_1(prices: List[int]) -> int:
    stack = []  # mono decreasing stack to keep the lowest price on top of the stack
    profit = 0
    for p in prices:
        if not stack or p < stack[-1]:
            stack.append(p)
        else:
            profit = max(profit, p - stack[-1])

    return profit


def max_profit_2(prices: List[int]) -> int:
    profit = 0
    buy_value = prices[0]
    for value in prices[1:]:
        if value > buy_value:
            profit = max(profit, value - buy_value)
        else:
            buy_value = value
    return profit


def max_profit_3(prices: List[int]) -> int:
    profit = 0
    buy_value = prices[0]
    for value in prices[1:]:
        if value < buy_value:
            buy_value = value
        elif profit < value - buy_value:
            profit = value - buy_value
    return profit
