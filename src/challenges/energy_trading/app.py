import logging
import pprint

from environs import Env

from challenges.energy_trading.data_access import load_energy_data
from challenges.energy_trading.trading_strategy import (
    calculate_daily_trading_strategy,
    max_profit_explicit_actions_pandas,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

if __name__ == "__main__":
    env = Env()
    env.read_env()

    energy_prices = load_energy_data(env("ENERGY_TRADING_DATA"))
    results = calculate_daily_trading_strategy(energy_prices)
    pprint.pp(results["2022-01-01"])

    print()
    results = max_profit_explicit_actions_pandas(energy_prices)
    pprint.pp(results.head(10))
