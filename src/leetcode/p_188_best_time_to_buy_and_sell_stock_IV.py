"""https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/description/

188. Best Time to Buy and Sell Stock IV
Hard

You are given an integer array prices where prices[i] is the price of
a given stock on the ith day, and an integer k.

Find the maximum profit you can achieve. You may complete at most k transactions:
i.e. you may buy at most k times and sell at most k times.

Note: You may not engage in multiple transactions simultaneously
(i.e., you must sell the stock before you buy again).



Example 1:
Input: k = 2, prices = [2,4,1]
Output: 2
Explanation: Buy on day 1 (price = 2) and sell on day 2 (price = 4), profit = 4-2 = 2.

Example 2:
Input: k = 2, prices = [3,2,6,5,0,3]
Output: 7
Explanation: Buy on day 2 (price = 2) and sell on day 3 (price = 6), profit = 6-2 = 4.
Then buy on day 5 (price = 0) and sell on day 6 (price = 3), profit = 3-0 = 3.


Constraints:

1 <= k <= 100
1 <= prices.length <= 1000
0 <= prices[i] <= 1000
"""

from typing import List, Tuple


def max_profit_with_k_transactions(prices: List[int], k: int) -> int:
    """Calculate the maximum profit that can be achieved from up to k stock transactions.

    This function is optimized to handle scenarios with a high number of transactions (k) by
    exploiting every increment in stock prices, ensuring profitability from every possible
    upward price movement.

    Args:
    ----
        prices (List[int]): List of daily stock prices.
        k (int): Maximum number of allowable transactions.

    Returns:
    -------
        int: Maximum profit achievable with up to k transactions.

    Example:
    -------
        >>> max_profit_k_transactions([3, 5, 1, 4, 6], 3)
        7

    Detailed Strategy:
    The idea is to create a table dp, where dp[transaction][day] represents the maximum profit achievable
    using up to t transactions by the dth day.

    - **Unlimited Transactions**:
        When `k` is at least half the number of days, the function assumes an infinite number of transactions.
        It then sums the profits of all ascending adjacent day pairs, capturing every profitable opportunity.
        When k is at least half the number of days, it effectively means we can make as
        many transactions as we want (since each transaction needs at least two days).
        Hence, we sum up all profitable adjacent days.

    - **Fluctuating Markets**:
        The strategy performs effectively even when prices fluctuate. It engages in a transaction at each
        opportunity where a price increment from one day to the next allows for a profit.
        For instance, in the price sequence [3, 5, 1, 4, 6], the function:
        1. Buys at 3 and sells at 5 (profit = 2).
        2. Skips the drop from 5 to 1 (no transaction).
        3. Buys at 1 and sells at 4 (profit = 3).
        4. Buys at 4 and sells at 6 (profit = 2).
        Total profit from these fluctuations is 2 + 3 + 2 = 7.

        This approach accumulates profits from all such upward movements,
        irrespective of the general trend of the market. Whether the prices overall
        are trending up, down, or fluctuating, as long as there are individual
        increments from one day to the next, they can be exploited for profit.


    Potential Issues:
        - **Overfitting to Data**: This strategy might not translate directly to real-time trading where
          future prices are unknown, and market conditions are unpredictable.
        - **Transaction Costs**: In real-world scenarios, transaction costs can reduce profitability.
          The strategy should be adjusted to consider costs, ensuring transactions only when the
          expected profit exceeds these costs.

    """
    n = len(prices)
    if n < 2 or k <= 0:
        # no transaction is possible
        return 0

    if k >= n // 2:
        # If k is large enough, solve as infinite transactions problem.
        return sum(max(prices[i] - prices[i - 1], 0) for i in range(1, n))

    # DP table initialization.
    # - each column represents a day, from day 0 to day n-1
    # - each row represents the number of transactions used to achieve the profit
    #   stored in that row. For instance, dp[0][d] would represent the
    #   maximum profit achievable by day d with zero transactions, which is
    #   important for initializing the basis and providing a starting point for no transactions
    # - each entry dp[t][d] holds the maximum profit that can be achieved by the end of day d
    #   using up to t transactions.
    dp = [[0] * n for _ in range(k + 1)]

    # Fill the DP table
    for t in range(1, k + 1):
        max_profit_after_cost = -prices[0]  # hypothetical buy action had occurred at the very first day.
        for day in range(1, n):
            # profit if no transaction is made on the current day.
            # Carries forward the maximum profit obtained up to the
            # previous day using the same number of transactions
            no_action_profit = dp[t][day - 1]

            # sell action, where:
            #   Selling the stock on day gou gain prices[day].
            #   After buying it on an earlier optimal day, the best earlier buy
            #   is encapsulated within max_profit_after_cost, which contains
            #   the optimal purchase scenario adjusted against the profits
            #   up to the previous transactions minus the price of the stock
            #   on that best buy day.
            sell_action_profit = prices[day] + max_profit_after_cost

            dp[t][day] = max(no_action_profit, sell_action_profit)

            # Adjust the maximum potential profit so far, by computing
            # what the profit would be if a transaction (buy followed by a sell)
            # were concluded on the previous day, and the new buy were
            # initiated today (day).
            # The variable potential_new_buy_profit represents the potential profit
            # if a new buy transaction were initiated on the current day,
            # following the last transaction that was completed the day before.
            potential_new_buy_profit = dp[t - 1][day - 1] - prices[day]
            max_profit_after_cost = max(max_profit_after_cost, potential_new_buy_profit)

    # Return the maximum profit achievable with k transactions by the last day
    return dp[k][n - 1]


def list_transaction_maximizing_profit(prices: List[int], k: int) -> List[Tuple[int, int, int]]:
    """Transaction Tracking: Whenever a sell operation (that maximizes the profit) is considered,
    the associated transaction is recorded, including both the day of purchase and the day of sale,
    as well as the profit from that specific transaction.
    """
    n = len(prices)
    if n < 2 or k <= 0:
        return []

    # If k is large enough, solve as infinite transactions problem
    if k > n // 2:
        current_transactions = []
        for i in range(1, n):
            transacation_profit = prices[i] - prices[i - 1]
            if transacation_profit > 0:
                current_transactions.append((i - 1, i, transacation_profit))
        return current_transactions

    # DP table initialization
    # stores tuples containing the maximum profit and a list of transactions
    # that led to that profit. Each transaction can be represented as
    # a tuple (buy_day, sell_day, profit).
    dp = [[(0, []) for _ in range(n)] for _ in range(k + 1)]

    # Fill the DP table
    for t in range(1, k + 1):
        max_profit_before_buying = -prices[0]
        for day in range(1, n):
            # Option 1: Don't transact on day d
            current_profit, current_transactions = dp[t][day - 1]

            # Option 2: Sell on day d
            # Calculate profit if selling on day d after buying on a previous day
            potential_profit = prices[day] + max_profit_before_buying

            if potential_profit > current_profit:
                new_transactions = updated_transaction_list(prices, t, day, dp)
                dp[t][day] = (potential_profit, new_transactions)
            else:
                dp[t][day] = (current_profit, current_transactions)

            # Update the max_profit_before_buying for the next day
            potential_new_buy_profit = dp[t - 1][day - 1][0] - prices[day]
            max_profit_before_buying = max(max_profit_before_buying, potential_new_buy_profit)

    return dp[k][n - 1][1]


def updated_transaction_list(prices, current_transaction, sell_day, dp) -> List[Tuple[int, int, int]]:
    """Update the transaction list with the optimal buy day for a given sell day.

    Args:
    ----
        prices (List[int]): List of daily stock prices.
        t (int): Current transaction count.
        sell_day (int): Current day considered for selling.
        dp (List[List[Tuple[int, List[Tuple[int, int, int]]]]]): DP table storing max profits and transactions.

    Returns:
    -------
        List[Tuple[int, int, int]]: Updated list of transactions including the optimal new transaction.

    """
    # Initialize with the previous day as the best day to buy
    # (this will change if a better day is found)
    best_buy_day = sell_day - 1

    max_profit_after_cost = dp[current_transaction - 1][sell_day - 1][0] - prices[sell_day - 1]

    # Iterate over days from sell_day-1 to 0
    for day in reversed(range(sell_day)):
        profit_if_bought_today = dp[current_transaction - 1][day][0] - prices[day]
        if profit_if_bought_today > max_profit_after_cost:
            max_profit_after_cost = profit_if_bought_today
            best_buy_day = day

    profit_from_transaction = prices[sell_day] - prices[best_buy_day]
    new_transaction = (best_buy_day, sell_day, profit_from_transaction)
    previous_transactions = dp[current_transaction - 1][best_buy_day][1]

    return previous_transactions + [new_transaction]


def aggregate_profit_from_transactions(transactions: List[Tuple[int, int, int]]) -> int:
    return sum([profit for _, _, profit in transactions])


if __name__ == "__main__":
    test_cases = [
        ([], 1, 0),  # [] Empty price list
        ([5, 5, 5, 5], 2, 0),  # [] No profit possible
        ([5, 5, 5, 5], 0, 0),  # [] No transaction allowed
        ([5, 4, 3, 2, 1], 2, 0),  # [] Monotonically decreasing prices
        ([2, 4, 1], 2, 2),  # [(0, 1, 2)]
        ([3, 3, 5, 0, 0, 3, 1, 4], 2, 6),  # [(4, 5, 3), (6, 7, 3)]
        # matching transactions available
        ([1, 2, 3, 4, 5], 2, 4),  # [(0, 4, 4)] Monotonically increasing prices ERR
        ([2, 1, 2, 0, 1], 2, 2),  # [(1, 2, 1), (3, 4, 1)]) non-overlapping transactions
        ([3, 2, 6, 5, 0, 3], 2, 7),  # [(1, 2, 4), (3, 5, 3)]  non-overlapping transactions
        ([1, 5, 2, 8, 3, 10], 3, 17),  # [(0, 1, 4), (2, 3, 6), (4, 5, 7)]) non-overlapping transactions
        ([1, 2, 4, 2, 5, 7, 2, 4, 9, 0], 2, 13),  # [(0, 5, 6), (6, 8, 7)]  overlapping transactions
        # more transactions avalable than requested
        (
            [7, 1, 5, 2, 4, 8, 3, 10, 11, 12, 15, 14, 4, 2, 0],  # [(1, 10, 14)] Multiple transactions
            1,
            14,
        ),
        (
            [7, 1, 5, 2, 4, 8, 3, 10, 11, 12, 15, 14, 4, 2, 0],  # [(1, 5, 7), (6, 10, 12)] Multiple transactions
            2,
            19,
        ),
        (
            [7, 1, 5, 2, 4, 8, 3, 10, 11, 12, 15, 14, 4, 2, 0],  # [(1, 2, 4), (3, 5, 6), (6, 10, 12)] Mult transactions
            3,
            22,
        ),
        (
            [7, 1, 5, 2, 4, 8, 3, 10, 11, 12, 15, 14, 4, 2, 0, 1, 5, 7, 4, 0],  # infinit transactions
            9999,
            29,
            # [
            #     (1, 2, 4),
            #     (3, 4, 2),
            #     (4, 5, 4),
            #     (6, 7, 7),
            #     (7, 8, 1),
            #     (8, 9, 1),
            #     (9, 10, 3),
            #     (14, 15, 1),
            #     (15, 16, 4),
            #     (16, 17, 2),
            # ]
        ),
    ]

    for input_arr, k, expected in test_cases:
        print(f"Prices list: {input_arr}")
        max_k_prof = max_profit_with_k_transactions(input_arr, k)
        buy_sell_trans = list_transaction_maximizing_profit(input_arr, k)
        total_profit = aggregate_profit_from_transactions(buy_sell_trans)
        print(f"\texpected '{expected}' | actuals: ('{total_profit}', '{max_k_prof}') | transactions'{buy_sell_trans}'")
        print()

    # TODO: arrange the transaction as a list of [buy, sell, skip]
