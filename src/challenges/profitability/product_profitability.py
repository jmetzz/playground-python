import pprint
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import more_itertools
from environs import Env

from challenges.profitability.data_access import DBReader
from challenges.profitability.entities import Product

env = Env()
env.read_env()


def products_profit(sales: List[List[int]]) -> Dict:
    """The most basic possible

    The input matrix of numbers and each column represents either ids or currency
    Ex:
    ---------------------------------------------------------------
    | sales_id | prod_id | unit_cost | unit_sales_price | quantity |
    ---------------------------------------------------------------

    """
    product_profit_registry = defaultdict(int)

    for _, prod_id, unit_cost, unit_sales_price, quantity in sales:
        product_profit_registry[prod_id] += (unit_sales_price - unit_cost) * quantity

    product_profit_registry = sorted(product_profit_registry.items(), key=lambda item: item[1])
    return product_profit_registry


def least_and_most_profitable_products(sales: List[List[int]]) -> Dict:
    """The most basic possible

    The input matrix of numbers and each column represents either ids or currency
    Ex:
    ---------------------------------------------------------------
    | sales_id | prod_d | unit_cost | unit_sales_price | quantity |
    ---------------------------------------------------------------

    """
    product_profit_registry = defaultdict(int)

    for _, prod_id, unit_cost, unit_sales_price, quantity in sales:
        product_profit_registry[prod_id] += (unit_sales_price - unit_cost) * quantity

    product_profit_registry = sorted(product_profit_registry.items(), key=lambda item: item[1])
    lowest_profit = product_profit_registry[0]
    highest_profit = product_profit_registry[-1]

    return build_response(highest_profit, lowest_profit)


def build_response(highest_profit_product: Tuple, lowest_profit_product: Tuple, products_details: Dict = None) -> Dict:
    return {
        "most_profitable": {
            "prod_id": highest_profit_product[0],
            "profit": highest_profit_product[1],
            "details": products_details.get(highest_profit_product[0]) if products_details else None,
        },
        "least_profitable": {
            "prod_id": lowest_profit_product[0],
            "profit": lowest_profit_product[1],
            "details": products_details.get(lowest_profit_product[0]) if products_details else None,
        },
    }


def get_least_and_most_profitable_products(db_reader, products: List[int], back_period: int = 30) -> Dict:
    """
    Assuming the input are valid already.

    Args:
        products: list of products ids sold by companies
        period: the number of days to consider for the calculation
    Return:
        least and most profitable product
    """

    sales = db_reader.list_sales(products, back_periods=back_period, period_unit="days")
    if not sales:
        # Consider raise a Error to bubble up the issue.
        # This decision should align with how the rest of the application
        # handles similar situations.
        return {}

    product_profit_registry = defaultdict(int)
    for sale in sales:
        product_profit_registry[sale.product_id] += (sale.unit_sales_price - sale.unit_cost) * sale.quantity
    product_profit_registry = sorted(product_profit_registry.items(), key=lambda item: item[1])  # List[(id, profit)]

    lowest_profit = product_profit_registry[0]
    highest_profit = product_profit_registry[-1]

    products_details = {p.id: p for p in db_reader.product_detail([lowest_profit[0], highest_profit[0]])}
    return build_response(highest_profit, lowest_profit, products_details)


def reduce_least_and_most_profitable_products(db_reader, sales: List[Product]):
    sales_profits = more_itertools.map_reduce(
        iterable=sales,
        keyfunc=lambda x: x.product_id,
        valuefunc=lambda x: (x.unit_sales_price - x.unit_cost) * x.quantity,
        reducefunc=sum,
    )
    lowest_profit, highest_profit = more_itertools.minmax(sales_profits.items(), key=lambda x: x[1])
    # how to solve the ties with minmax approach above?

    products_details = {p.id: p for p in db_reader.product_detail([lowest_profit[0], highest_profit[0]])}
    return build_response(highest_profit, lowest_profit, products_details)


def df_get_least_and_most_profitable_products(db_reader, products: List[int], back_periods: int = 30) -> Dict:
    sales_df = db_reader.list_sales_details(products, back_periods=back_periods, period_unit="days", as_df=True)
    if sales_df.empty:
        # Consider raise a Error to bubble up the issue.
        # This decision should align with how the rest of the application
        # handles similar situations.
        return {}

    # Make use of vectorized feature from pandas:
    sales_df["profit"] = (sales_df["unit_sales_price"] - sales_df["unit_cost"]) * sales_df["quantity"]

    # Group by product_id and sum the profits
    profit_by_product = sales_df.groupby("product_id")["profit"].sum()

    # Find the index of the highest and lowest profit
    highest_p_idx = profit_by_product.idxmax()
    lowest_p_idx = profit_by_product.idxmin()

    products_details = {p.id: p for p in db_reader.product_detail([lowest_p_idx, highest_p_idx])}
    return build_response(
        (highest_p_idx, profit_by_product.loc[highest_p_idx]),
        (lowest_p_idx, profit_by_product.loc[lowest_p_idx]),
        products_details,
    )


def get_least_and_most_profitable_products_per_geolocation(products: List[int], geo: str, period: int = 30) -> Dict:
    raise NotImplementedError()

    lowest_profit = None
    highest_profit = None

    return {
        "most_profitable": {
            "prod_id": highest_profit[0],
            "profit": highest_profit[1],
            # "details": products_details[highest_profit[0]],
        },
        "least_profitable": {
            "prod_id": lowest_profit[0],
            "profit": lowest_profit[1],
            # "details": products_details[lowest_profit[0]],
        },
    }


def get_least_and_most_profitable_products_per_provider(products: List[int], provider: str, period: int = 30) -> Dict:
    raise NotImplementedError()
    lowest_profit = None
    highest_profit = None

    return {
        "most_profitable": {
            "prod_id": highest_profit[0],
            "profit": highest_profit[1],
            # "details": products_details[highest_profit[0]],
        },
        "least_profitable": {
            "prod_id": lowest_profit[0],
            "profit": lowest_profit[1],
            # "details": products_details[lowest_profit[0]],
        },
    }


def solution(lines: List[str]):
    company_sales = {}  # company_id -> { prod_id : float}
    for line in lines:
        product_id, company_id, cost, revenue = line.split(",")
        if company_id not in company_sales:
            company_sales[company_id] = defaultdict(float)
        company_sales[company_id][product_id] += float(revenue) - float(cost)
    companies = company_sales.keys()

    answer = []
    for company in companies:
        products_profit = sorted(company_sales[company].items(), key=lambda item: item[1])
        answer.append((company, products_profit[-1][0], products_profit[0][0]))

    return answer


def main():
    lines = [
        # product_id, company_id, cost, revenue
        "1,1,0.2,0.5",
        "2,1,0.25,0.1",
        "3,1,0.25,0.25",
        "4,2,1,1",
    ]
    print(solution(lines))


if __name__ == "__main__":
    # the real interview test:
    main()

    pp = pprint.PrettyPrinter()

    """
    ---------------------------------------------------------------
    | sales_id | prod_d | unit_cost | unit_sales_price | quantity |
    ---------------------------------------------------------------
    """
    sales_data = [
        # prod 1 sales
        [1, 1, 1, 2, 3],
        [2, 1, 1, 2, 3],
        [3, 1, 1, 5, 1],
        [4, 1, 1, 1, 10],
        # prod 2 sales
        [5, 2, 10, 12, 3],
        [6, 2, 10, 11, 3],
        [7, 2, 10, 15, 1],
        [8, 2, 10, 5, 10],
        [9, 2, 10, 14, 7],
        [10, 2, 10, 12, 3],
        [11, 2, 10, 17, 1],
        # prod 3 sales
        [12, 3, 8, 12, 8],  # 32
        [13, 3, 8, 15, 7],  # 49
        [14, 3, 8, 20, 1],  # 12
        [15, 3, 8, 19, 4],  # 44 --> 137
        # prod 4 sales
        [16, 4, 20, 21, 1],  # 1
        [17, 4, 20, 18, 12],  # -24
        [18, 4, 20, 15, 50],  # -250
        [19, 4, 20, 21, 4],  # 4 -->-269
        # returns
        [18, 4, 20, 15, -5],  # returning 5 units bought at 15 per unit.
    ]

    print("Product's profit:")
    print(products_profit(sales_data))

    print()
    print("Least and most profitable:")
    pp.pprint(least_and_most_profitable_products(sales_data))

    data_path = Path(env("RESOURCES_PATH"), env("AWESOME_EXPERIENCES_DATA_FILE"))
    db_path = Path(env("RESOURCES_PATH"), env("AWESOME_EXPERIENCES_DB_PATH"))
    db_reader = DBReader(db_path)
    # print(db_reader.product_detail([1, 2, 3]))
    # print("----")
    # print(db_reader.list_sales([1, 2, 3], back_periods=30, period_unit="days"))

    print("\n\n=====================")
    print("FROM db:")
    pp.pprint(get_least_and_most_profitable_products(db_reader, [1, 2, 3], 10))

    print("\n\n=====================")
    print("Map reduce:")
    sales = db_reader.list_sales([1, 2, 3], back_periods=10, period_unit="days")
    pp.pprint(reduce_least_and_most_profitable_products(db_reader, sales))

    pp.pprint(df_get_least_and_most_profitable_products(db_reader, [1, 2, 3], back_periods=10))
