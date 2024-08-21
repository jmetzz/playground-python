import random
import sqlite3
from datetime import datetime, timedelta
from typing import Any, List, Tuple

import pandas as pd

from challenges.profitability.entities import Product, Sales


class DBReader:
    def __init__(self, database_path):
        self._db_path = database_path

    def run_query(self, query: str, params: Tuple = ()) -> List[Any]:
        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                return cursor.fetchall()
        except sqlite3.Error as error:
            print("Error while executing the query:", error)

    def run_query_as_df(self, query: str):
        with sqlite3.connect(self._db_path) as conn:
            return pd.read_sql_query(query, conn)

    def product_detail(self, product_ids: List[int]) -> List[Product]:
        placeholders = ", ".join(["?"] * len(product_ids))
        query = f"SELECT * FROM product WHERE ID IN ({placeholders})"
        query_result = self.run_query(query, product_ids)
        return [Product(*row) for row in query_result]

    def list_sales_details(
        self,
        product_ids: List[int],
        ref_date: StopAsyncIteration = None,
        back_periods: int = 1,
        period_unit: str = "month",
        as_df: bool = False,
    ):
        end_date = datetime.strptime(ref_date, "%Y-%m-%d") if ref_date else datetime.now()
        start_date = f"date('{end_date.strftime("%Y-%m-%d")}', '-{back_periods} {period_unit}')"

        query = f"""
        SELECT 
            s.id "sales_id", 
            p.id "product_id", 
            p.name "product", 
            sp.id "provider_id", 
            p.cost "unit_cost", 
            s.unit_price "unit_sales_price",
            s.quantity,
            sp.name "service_provider", 
            s.sale_date, 
            sp.geo_id
        FROM sales s
            right join product p on p.id = s.product_id
            join service_provider sp on s.service_provider_id = sp.id
        WHERE
            p.id in {tuple(product_ids)}
            and s.sale_date BETWEEN {start_date} AND '{end_date.strftime("%Y-%m-%d")}'
        ORDER BY s.sale_date;
        """
        return self.run_query_as_df(query) if as_df else self.run_query(query)

    def list_sales(
        self,
        product_ids: List[int],
        ref_date: StopAsyncIteration = None,
        back_periods: int = 1,
        period_unit: str = "month",
    ) -> List[Product]:
        end_date = datetime.strptime(ref_date, "%Y-%m-%d") if ref_date else datetime.now()
        start_date = f"date('{end_date.strftime("%Y-%m-%d")}', '-{back_periods} {period_unit}')"

        query = f"""
        SELECT 
            s.id "sales_id",
            p.id "product_id", 
            p.name "product", 
            p.cost "unit_cost", 
            s.unit_price "unit_sales_price",
            s.quantity
        FROM sales s
            right join product p on p.id = s.product_id
        WHERE
            p.id in {tuple(product_ids)}
            and s.sale_date BETWEEN {start_date} AND '{end_date.strftime("%Y-%m-%d")}'
        ORDER BY s.sale_date;
        """
        query_result = self.run_query(query)
        return [Sales(*row) for row in query_result]


def create_database(db_path: str, sql_data_path: str):
    """
    Create a SQLite database from an SQL file.
    """
    # Connect to the SQLite database. This will create the database file if it does not exist.
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Read the SQL commands from the .sql file
    with open(sql_data_path) as sql_file:
        sql_script = sql_file.read()

    # Execute the SQL commands
    try:
        cursor.executescript(sql_script)
        conn.commit()
        print("Database has been created and populated with data successfully.")
    except sqlite3.Error as error:
        print("Error while executing the SQL script:", error)
        conn.rollback()  # Rollback any changes if something goes wrong
    finally:
        # Close the database connection
        cursor.close()
        conn.close()


def generate_sales_data(db_file, num_sales, dry_run: bool = False):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Retrieve the list of product and service provider IDs from their respective tables
    cursor.execute("SELECT id FROM product")
    product_ids = [row[0] for row in cursor.fetchall()]

    cursor.execute("SELECT id FROM service_provider")
    service_provider_ids = [row[0] for row in cursor.fetchall()]

    # Generate sample sales data
    for _ in range(num_sales):
        # Choose a random product ID and service provider ID
        product_id = random.choice(product_ids)
        service_provider_id = random.choice(service_provider_ids)

        # Generate a random sale date within the past year
        sale_date = datetime.now() - timedelta(days=random.randint(0, 365))

        # Generate a random quantity sold (between 1 and 10)
        quantity = random.randint(1, 10)

        # Retrieve the unit price of the product from the product table
        cursor.execute("SELECT cost FROM product WHERE id=?", (product_id,))
        unit_cost = cursor.fetchone()[0]

        # Calculate the total sale amount
        unit_price = round(unit_cost * (1 + random.uniform(0.1, 0.7)), 2)

        # Insert the generated sales data into the sales table
        item = (product_id, service_provider_id, sale_date.strftime("%Y-%m-%d"), quantity, unit_price)

        if not dry_run:
            cursor.execute(
                "INSERT INTO sales (product_id, service_provider_id, sale_date, quantity, unit_price)"
                "VALUES (?, ?, ?, ?, ?)",
                item,
            )
        else:
            print(item)

    # Commit changes and close the database connection
    conn.commit()
    conn.close()


def export_sqlite_db_to_sql(db_file, output_sql_file):
    import subprocess

    # Command to export SQLite database to a SQL file
    command = f"sqlite3 {db_file} .dump > {output_sql_file}"

    # Execute the command
    subprocess.run(command, shell=True)
