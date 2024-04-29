from typing import List, Tuple


def max_profit_unlimited_transactions_recursive(prices: List[float]) -> float:
    def _recurse(price_rates: List[float], start, end):
        if end <= start:
            return 0
        profit = 0
        # actions each hour
        # - charge, if battery is not charged
        # - discharge, if batery is charged
        # - do nothing, regardless of the batery state
        for i in range(start, end, 1):
            for j in range(i + 1, end + 1, 1):
                if price_rates[j] > price_rates[i]:
                    curr_profit = (
                        price_rates[j]
                        - price_rates[i]
                        + _recurse(price_rates, start, i - 1)
                        + _recurse(price_rates, j + 1, end)
                    )
                    profit = max(profit, curr_profit)
        return profit

    return _recurse(prices, 0, len(prices) - 1)


def max_profit_unlimited_transactions_1(prices: List[float]) -> float:
    # accumulate the profit on upward trend from one timestamp to the next
    profit = 0
    price = prices[0]

    for idx in range(1, len(prices)):
        if prices[idx] > price:
            profit += prices[idx] - price

        price = prices[idx]
    return profit


def max_profit_unlimited_transactions_2(prices: List[int]) -> int:
    n = len(prices)
    if n < 2:  # no transaction is possible
        return 0
    return sum(max(prices[i] - prices[i - 1], 0) for i in range(1, n))


def max_profit_with_k_transactions(prices: List[int], k: int) -> int:
    n = len(prices)
    if n < 2 or k <= 0:
        # no transaction is possible
        return 0

    if k >= n // 2:
        # If k is large enough, solve as infinite transactions problem.
        return sum(max(prices[i] - prices[i - 1], 0) for i in range(1, n))

    dp = [[0] * n for _ in range(k + 1)]

    # Fill the DP table
    for t in range(1, k + 1):
        max_profit_after_cost = -prices[0]  # hypothetical buy action had occurred at the very first day.
        for day in range(1, n):
            no_action_profit = dp[t][day - 1]

            sell_action_profit = prices[day] + max_profit_after_cost

            dp[t][day] = max(no_action_profit, sell_action_profit)

            potential_new_buy_profit = dp[t - 1][day - 1] - prices[day]
            max_profit_after_cost = max(max_profit_after_cost, potential_new_buy_profit)

    return dp[k][n - 1]


def list_transaction_maximizing_profit(prices: List[int], k: int) -> List[Tuple[int, int, int]]:
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


if __name__ == "__main__":
    hourly_prices = [
        50.05,
        41.33,
        43.22,
        45.46,
        37.67,
        39.70,
        40.59,
        43.26,
        49.66,
        70.05,
        76.79,
        84.10,
        94.74,
        96.80,
        97.17,
        101.00,
        126.60,
        149.97,
        146.33,
        140.28,
        121.88,
        102.61,
        97.46,
        85.16,
    ]

    print(f"Expected: {149.97 - 37.67 + 45.46 - 41.33:.2f}")
    print(f"\t{max_profit_unlimited_transactions_recursive(hourly_prices)}")
    print(f"\t{max_profit_unlimited_transactions_1(hourly_prices)}")
    print(f"\t{max_profit_unlimited_transactions_2(hourly_prices)}")
    print(f"\t{max_profit_with_k_transactions(hourly_prices, k=len(hourly_prices))}")
