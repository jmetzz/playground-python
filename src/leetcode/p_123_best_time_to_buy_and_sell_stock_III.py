"""https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/

123. Best Time to Buy and Sell Stock III
Hard

You are given an array prices where prices[i] is the price of a given stock on the ith day.

Find the maximum profit you can achieve. You may complete at most two transactions.

Note: You may not engage in multiple transactions simultaneously
(i.e., you must sell the stock before you buy again).



Example 1:

Input: prices = [3,3,5,0,0,3,1,4]
Output: 6
Explanation: Buy on day 4 (price = 0) and sell on day 6 (price = 3), profit = 3-0 = 3.
Then buy on day 7 (price = 1) and sell on day 8 (price = 4), profit = 4-1 = 3.

Example 2:
Input: prices = [1,2,3,4,5]
Output: 4
Explanation: Buy on day 1 (price = 1) and sell on day 5 (price = 5), profit = 5-1 = 4.
Note that you cannot buy on day 1, buy on day 2 and sell them later,
as you are engaging multiple transactions at the same time. You must sell before buying again.

Example 3:
Input: prices = [7,6,4,3,1]
Output: 0
Explanation: In this case, no transaction is done, i.e. max profit = 0.


Constraints:

1 <= prices.length <= 10**5
0 <= prices[i] <= 10**5
"""

from heapq import heappop, heappush
from typing import Dict, List, Tuple


def max_profit_heap(prices: List[int]) -> int:
    """This function does not solve the problem!

    It only solves local sub-problems.
    """
    transactions_heap = []  # max heap: profit
    stack = [(0, 0)]  # (start_idx, curr_idx)

    for idx in range(1, len(prices)):
        if prices[idx] >= prices[idx - 1]:
            # continue expanding the profit since we are at a upwards trend
            start_idx, _ = stack[-1]
        else:
            # realise profit since the trend has inverted downwards
            start_idx, prev_idx = stack.pop()
            profit = prices[prev_idx] - prices[start_idx]

            # use a heap to keep only the two most profitable transactions
            if profit > 0:
                profit = -profit
                if not transactions_heap or len(transactions_heap) < 2:
                    heappush(transactions_heap, profit)
                elif transactions_heap[0] > profit:
                    heappop(transactions_heap)
                    heappush(transactions_heap, profit)

            # remove these data points from the stack?
            while stack and stack[-1][1] != start_idx:
                stack.pop()

            # reset the start_idx for the current lower value
            start_idx = idx

        stack.append((start_idx, idx))

    # calculate the profit of a potential transaction still on stack
    # This happens if at some point in the prices array
    # the values become monotonicaly increasing, ie, [1,2,3,4,5,7,9]
    # and thus, the transaction calculation block above will never be executed
    if stack:
        start_idx, end_idx = stack[-1]
        heappush(transactions_heap, -(prices[end_idx] - prices[start_idx]))

    qtt = 0
    total_profit = 0
    while transactions_heap and qtt < 2:
        total_profit += heappop(transactions_heap)
        qtt += 1

    return -total_profit


def max_profit_list_transactions(prices: List[int]) -> List[Tuple]:
    """This function does not solve the problem!

    It only solves local sub-problems.
    """
    transactions_heap = []  # max heap -> (profit, start_idx, end_idx)
    stack = [(0, 0)]  # (start_idx, curr_idx)

    for idx in range(1, len(prices)):
        if prices[idx] >= prices[idx - 1]:
            # continue expanding the profit since we are at a upwards trend
            start_idx, _ = stack[-1]
        else:
            # realise profit since the trend has inverted downwards
            start_idx, prev_idx = stack.pop()
            profit = prices[prev_idx] - prices[start_idx]
            transaction = -profit, start_idx, prev_idx

            # use a heap to keep only the two most profitable transactions
            if profit > 0:
                if not transactions_heap or len(transactions_heap) < 2:
                    heappush(transactions_heap, transaction)
                elif transactions_heap[0][0] > transaction[0]:
                    heappop(transactions_heap)
                    heappush(transactions_heap, transaction)

            # remove these data points from the stack?
            while stack and stack[-1][1] >= start_idx:
                stack.pop()

            # reset the start_idx for the current lower value
            start_idx = idx

        stack.append((start_idx, idx))

    # calculate the profit of a potential transaction still on stack
    # This happens if at some point in the prices array
    # the values become monotonicaly increasing, ie, [1,2,3,4,5,7,9]
    # and thus, the transaction calculation block above will never be executed
    if stack:
        start_idx, end_idx = stack[-1]
        profit = prices[end_idx] - prices[start_idx]
        if profit > 0:
            heappush(transactions_heap, (-profit, start_idx, end_idx))

    qtt = 0
    answer = []
    while transactions_heap and qtt < 2:
        profit, s, e = heappop(transactions_heap)
        answer.append((-profit, s, e))
        qtt += 1
    return answer


def max_profit_two_segments(prices: List[int]) -> List[Tuple]:
    """Max profit with at most two transactions"""
    if not prices:
        return 0

    n = len(prices)
    # Initialize left profits and right profits arrays
    left_profit, right_profit = [0] * n, [0] * n

    min_price_in_prefix = prices[0]
    for i in range(1, n):
        profit = prices[i] - min_price_in_prefix
        left_profit[i] = max(left_profit[i - 1], profit)
        min_price_in_prefix = min(min_price_in_prefix, prices[i])

    max_price_in_suffix = prices[-1]
    for i in reversed(range(n - 1)):
        profit = max_price_in_suffix - prices[i]
        right_profit[i] = max(right_profit[i + 1], profit)
        max_price_in_suffix = max(max_price_in_suffix, prices[i])

    # Calculate the maximum profit possible
    answer = right_profit[0]  # This covers the case where the best profit involves no split
    for i in range(1, n - 1):
        answer = max(answer, left_profit[i] + right_profit[i + 1])
    return answer


def max_profit_two_segments_list_transactions(prices: List[int]) -> List[Tuple]:
    """To identify the days in which the transactions happen,
    we can extend the arrays to track days information.

    Instead of just tracking profits, also track the days on which the buy and sell happen.
    This means each entry in left_profit and right_profit could now be a tuple
    consisting of (profit, buy_day, sell_day).


    """
    if not prices:
        return 0

    n = len(prices)
    # default of no profit and no transactions.
    left_profit = [(0, None, None)] * n  # (profit, buy_day, sell_day)
    right_profit = [(0, None, None)] * n  # (profit, buy_day, sell_day).

    min_price_index = 0
    for i in range(1, n):
        if prices[i] - prices[min_price_index] > left_profit[i - 1][0]:
            left_profit[i] = (prices[i] - prices[min_price_index], min_price_index, i)
        else:
            left_profit[i] = left_profit[i - 1]
        if prices[i] < prices[min_price_index]:
            min_price_index = i

    max_price_index = n - 1
    for i in reversed(range(n - 1)):
        if prices[max_price_index] - prices[i] > right_profit[i + 1][0]:
            right_profit[i] = (prices[max_price_index] - prices[i], i, max_price_index)
        else:
            right_profit[i] = right_profit[i + 1]
        if prices[i] > prices[max_price_index]:
            max_price_index = i

    # Find the best combination
    max_profit = 0
    best_days = (None, None), (None, None)
    for i in range(n):
        potential_profit = left_profit[i][0] + right_profit[i][0]
        if potential_profit > max_profit:
            max_profit = potential_profit
            best_days = (left_profit[i][1], left_profit[i][2]), (right_profit[i][1], right_profit[i][2])

    transactions = []
    for start, end in best_days:
        if start is not None:
            transactions.append((prices[end] - prices[start], start, end))
    return max_profit, transactions


def max_profit_recursive(prices: List[int]) -> List[Tuple]:
    """ """

    def dfs(index: int, can_buy: bool, transactions_allowance: int):
        if index == n:
            return 0
        if transactions_allowance == 0:
            return 0

        profit = 0
        if can_buy:
            buy_profit = dfs(index + 1, not can_buy, transactions_allowance) - prices[index]
            skip_profit = dfs(index + 1, can_buy, transactions_allowance)
            profit = max(buy_profit, skip_profit)
        else:
            sell_profit = dfs(index + 1, not can_buy, transactions_allowance - 1) + prices[index]
            skip_profit = dfs(index + 1, can_buy, transactions_allowance)
            profit = max(sell_profit, skip_profit)

        return profit

    n = len(prices)
    return dfs(index=0, can_buy=True, transactions_allowance=2)


def max_profit_memoization(prices: List[int]) -> List[Tuple]:
    def dfs(index: int, can_buy: bool, transactions_allowance: int, cache: Dict):
        if index == n:
            return 0
        if transactions_allowance == 0:
            return 0

        if (index, can_buy, transactions_allowance) in cache:
            return cache[(index, can_buy, transactions_allowance)]

        profit = 0
        if can_buy:
            buy_profit = dfs(index + 1, not can_buy, transactions_allowance, cache) - prices[index]
            skip_profit = dfs(index + 1, can_buy, transactions_allowance, cache)
            profit = max(buy_profit, skip_profit)
        else:
            sell_profit = dfs(index + 1, not can_buy, transactions_allowance - 1, cache) + prices[index]
            skip_profit = dfs(index + 1, can_buy, transactions_allowance, cache)
            profit = max(sell_profit, skip_profit)
        cache[(index, can_buy, transactions_allowance)] = profit
        return profit

    n = len(prices)
    return dfs(index=0, can_buy=True, transactions_allowance=2, cache={})


def max_profit_tabulation(prices: List[int]) -> List[Tuple]:
    """Time Complexity O(N)
    Space Complexity O(N)
    """
    n = len(prices)

    # n + 1 elements, with 2 possible actions, and 2 possible transactions
    dp = [[[0 for _ in range(3)] for _ in range(2)] for _ in range(n + 1)]

    for idx in reversed(range(n)):
        for can_buy in range(2):
            for transaction in range(1, 3):
                profit = 0
                if can_buy == 1:
                    buy_profit = dp[idx + 1][0][transaction] - prices[idx]
                    skip_profit = dp[idx + 1][can_buy][transaction]
                    profit = max(buy_profit, skip_profit)
                else:
                    sell_profit = dp[idx + 1][1][transaction - 1] + prices[idx]
                    skip_profit = dp[idx + 1][can_buy][transaction]
                    profit = max(sell_profit, skip_profit)
                dp[idx][can_buy][transaction] = profit

    return dp[0][1][2]


def max_profit_space_optmized(prices: List[int]) -> List[Tuple]:
    n = len(prices)

    # 2 possible actions, and 2 possible transactions
    current = [[0 for _ in range(3)] for _ in range(2)]
    next_ = [[0 for _ in range(3)] for _ in range(2)]

    for idx in reversed(range(n)):
        for can_buy in range(2):
            for transaction in range(1, 3):
                profit = 0
                if can_buy == 1:
                    buy_profit = next_[0][transaction] - prices[idx]
                    skip_profit = next_[can_buy][transaction]
                    profit = max(buy_profit, skip_profit)
                else:
                    sell_profit = next_[1][transaction - 1] + prices[idx]
                    skip_profit = next_[can_buy][transaction]
                    profit = max(sell_profit, skip_profit)
                current[can_buy][transaction] = profit
            next_ = current
    return next_[1][2]


if __name__ == "__main__":
    test_cases = [
        ([7, 6, 4, 3, 1], 0),
        ([3, 3, 5, 0, 0, 3, 1, 4], 6),
        ([1, 2, 3, 4, 5], 4),
        ([2, 1, 2, 0, 1], 2),
        ([1, 2, 4, 2, 5, 7, 2, 4, 9, 0], 13),
    ]
    max_profit_funcs = [
        max_profit_heap,
        max_profit_two_segments,
        max_profit_recursive,
        max_profit_memoization,
        max_profit_tabulation,
        max_profit_space_optmized,
    ]

    for input_arr, expected in test_cases:
        print(f"expected '{expected}'")
        for func in max_profit_funcs:
            print(f"\t{func.__name__}: '{func(input_arr)}'")
        print()

    transaction_funcs = [max_profit_list_transactions, max_profit_two_segments_list_transactions]
    for input_arr, _ in test_cases:
        for func in transaction_funcs:
            print(f"{func.__name__}: '{func(input_arr)}'")
        print()
