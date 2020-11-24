import logging
from contextlib import contextmanager
from typing import Dict

import pandas as pd
from pandas.io.sql import DatabaseError as PandasDatabaseError
from psycopg2 import Error as Psycopg2BaseError
from psycopg2.pool import ThreadedConnectionPool

from py_oop.dataservice import config
from py_oop.dataservice.utils import timed

logger = logging.getLogger(__name__)

pool = ThreadedConnectionPool(
    minconn=config.MIN_CONN_COUNT,
    maxconn=config.MAX_CONN_COUNT,
    host=config.POSTGRES_HOST,
    port=config.POSTGRES_PORT,
    dbname=config.POSTGRES_DB,
    user=config.POSTGRES_USER,
    password=config.POSTGRES_PASS,
    connect_timeout=config.POSTGRES_TIMEOUT,
)


@contextmanager
def get_db_connection():
    """Context manager providing a connection from the connection pool.

    Ensures the connection is put back to the pool when finished and handles errors.

    Usage:

    with db.get_db_connection() as conn:
        ...
    """
    db_error = None

    try:
        conn = pool.get_conn()
        yield conn
    except (PandasDatabaseError, Psycopg2BaseError) as e:
        db_error = e
    finally:
        if db_error:
            # assume that we need to reinitialize connection pool when any db error happens
            pool.reset()
            raise db_error
        else:
            pool.put_conn(conn)
    conn = pool.getconn()
    conn.autocommit = True
    conn.readonly = True
    pre_queries = [
        "SET random_page_cost=1",
        "SET seq_page_cost=1",
        "SET effective_io_concurrency=100",
        "SET effective_cache_size='100GB'",
    ]
    with conn.cursor() as cursor:
        [cursor.execute(query) for query in pre_queries]

    yield conn
    conn.rollback()
    pool.putconn(conn)


@timed
def get_data_frame(query: str, params: Dict = None) -> pd.DataFrame:
    """Gets a connection and executes the query with the specified parameters.

    Returns a DataFrame with the results.
    """
    with get_db_connection() as conn:
        df = pd.read_sql_query(query, conn, params=params)

    return df
