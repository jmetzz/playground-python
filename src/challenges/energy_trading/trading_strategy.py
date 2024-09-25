import pandas as pd


def list_actions_maximixing_profit_1(prices):
    n = len(prices)
    if n < 2:
        return 0, ["skip" for _ in prices]  # No transactions possible

    actions = ["skip"] * n
    profit = 0
    buy_day = None  # To track if we have an ongoing transaction

    for idx in range(1, len(prices)):
        if prices[idx] > prices[idx - 1]:
            profit += prices[idx] - prices[idx - 1]
            buy_day = idx - 1 if buy_day is None else buy_day
            actions[buy_day] = "buy"
            actions[idx] = "sell"
            buy_day = None  # Reset for potential new transaction
        elif buy_day is not None:
            actions[buy_day] = "buy"  # Confirm the buy if not yet confirmed
            actions[idx - 1] = "sell"
            buy_day = None

    return profit, actions


def list_actions_maximixing_profit_2(prices: list[int]) -> tuple[int, list[str]]:
    n = len(prices)
    if n < 2:
        return 0, ["skip" for _ in prices]  # No transactions possible

    actions = ["skip"] * n
    profit = 0

    for idx in range(1, n):
        if prices[idx] > prices[idx - 1]:
            profit += prices[idx] - prices[idx - 1]
            actions[idx - 1] = "buy" if actions[idx - 1] != "buy" else actions[idx - 1]
            actions[idx] = "sell"

    # Fine-tune the actions to ensure logical trading:
    for idx in range(1, n):
        if actions[idx] == "sell" and actions[idx - 1] != "buy":
            actions[idx - 1] = "buy"  # Ensure there is a buy before a sell
        if actions[idx] == "buy" and idx < n - 1 and prices[idx + 1] <= prices[idx]:
            actions[idx] = "skip"  # Avoid buying if not selling for a profit later

    # Cleanup to ensure the first and last actions are logical
    if actions[0] == "sell":
        actions[0] = "skip"
    if actions[-1] == "buy":
        actions[-1] = "skip"

    return profit, actions


def max_profit_explicit_actions(prices: list[int]) -> tuple[int, list[str]]:
    n = len(prices)
    if n < 2:
        return 0, ["skip" for _ in prices]  # No transactions possible

    actions = ["skip"] * n
    profit = 0
    holding = False  # To track if we are currently holding a stock

    for i in range(1, n):
        if prices[i] > prices[i - 1]:
            profit += prices[i] - prices[i - 1]
            actions[i - 1] = "buy" if not holding else actions[i - 1]
            actions[i] = "sell"
            holding = False  # We sell at i
            if i < n - 1 and prices[i] < prices[i + 1]:  # Prepare to buy again if next price is higher
                actions[i] = "sell and buy"  # Indicate both actions at the same day
                holding = True
        else:
            if holding:
                actions[i] = "sell"
                holding = False
            if prices[i] < prices[i - 1] and i < n - 1 and prices[i] < prices[i + 1]:
                actions[i] = "buy"
                holding = True

    # Clean up the actions to ensure last action is not a buy without a sell
    if holding:
        actions[-1] = "sell"

    return profit, actions


def max_profit_explicit_actions_pandas(df: pd.DataFrame, group: bool = True) -> pd.DataFrame:
    # Calculate price differences
    df["Price Diff"] = df["Price"].diff()

    # Define actions based on price changes
    df["Action"] = "skip"
    df.loc[df["Price Diff"] > 0, "Action"] = "sell"  # Sell on positive change
    df.loc[df["Price Diff"] < 0, "Action"] = "buy"  # Buy on negative change

    # Calculate profit only when selling
    df["Profit"] = 0.0
    df.loc[df["Action"] == "sell", "Profit"] = df["Price Diff"]

    # Sum the profit and reshape the data by date
    if group:
        return df.groupby(df["start_time"].dt.date).agg(
            Total_Profit=("Profit", "sum"), Actions=("Action", lambda x: x.tolist())
        )

    return df


def calculate_profit_from_transaction_log(prices: list[int], transactions: list[str]) -> float:
    profit = 0
    first_buy = None
    for i, action in enumerate(transactions):
        match action:
            case "buy":
                first_buy = i
            case "sell":
                profit += prices[i] - prices[first_buy]
                first_buy = None
    return profit


def calculate_daily_trading_strategy(prices_df: pd.DataFrame, append: bool = False) -> dict:
    results = {}
    for day, group in prices_df.groupby(prices_df["start_time"].dt.date):
        prices = group["Price"].tolist()
        profit, actions = max_profit_explicit_actions(prices)
        results[str(day)] = {"strategy": actions, "profit": profit}
    return results


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

    profit, transactions_log = list_actions_maximixing_profit_1(hourly_prices)
    print(profit)
    print(transactions_log)
    print()

    profit, transactions_log = list_actions_maximixing_profit_2(hourly_prices)
    print(profit)
    print(transactions_log)
    print()

    profit, transactions_log = max_profit_explicit_actions(hourly_prices)
    print(profit)
    print(transactions_log)

    print(calculate_profit_from_transaction_log(hourly_prices, transactions_log))
