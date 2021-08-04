import random
import tracemalloc

import pandas as pd


def generate_data(factor: int) -> pd.DataFrame:
    random.seed(42)

    item_ids = [1000 + i for i in range(10 ** factor)]
    prices = [round(20 + random.random() * 50, 2) for _ in range(10 ** factor)]
    price_constants = [round((random.random() - 0.5) * 10, 2) for _ in range(10 ** factor)]
    units = [int(random.random() * 100) for _ in range(10 ** factor)]

    data_df = pd.DataFrame(
        data={
            "item_id": item_ids,
            "price": prices,
            "price_constant": price_constants,
            "units": units,
        }
    )
    return data_df


def trigger(factor: int) -> tuple:
    tracemalloc.start()
    _ = generate_data(factor)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return current, peak


def trigger_cases(factors: list):
    for f in factors:
        current, peak = trigger(f)
        print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")


trigger_cases([2, 3, 4, 5, 6, 7])
