import datetime
import logging
from contextlib import contextmanager
from typing import Dict

import config
import pandas as pd
from pandas.io.sql import DatabaseError as PandasDatabaseError
from psycopg2 import Error as Psycopg2BaseError
from psycopg2.pool import PoolError, ThreadedConnectionPool

logger = logging.getLogger(__name__)


class ConnectionPool:
    """Wraps psycopg2's ThreadedConnectionPool.

    Configures the connection pool based on our settings and adds some
    convenience methods.

    Entry points to the database are the `get_db_connection` context manager
    and the `run_query` function below.
    """

    def __init__(self):
        self._conn_pool = None

    @property
    def conn_pool(self):
        if self._conn_pool is None:
            logger.debug(
                {
                    "message": {
                        "action": "CONNECTION_POOL_CREATED",
                        "timestamp": datetime.datetime.now(),
                    }
                }
            )
            self._conn_pool = self.init_conn_pool()

        return self._conn_pool

    @staticmethod
    def init_conn_pool():
        return ThreadedConnectionPool(
            minconn=config.MIN_CONN_COUNT,
            maxconn=config.MAX_CONN_COUNT,
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT,
            dbname=config.POSTGRES_DB,
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASS,
            connect_timeout=config.POSTGRES_TIMEOUT,
        )

    def reset(self):
        # pool error gets raised when connection is already closed
        try:
            self.conn_pool.closeall()
        except PoolError:
            pass
        self._conn_pool = self.init_conn_pool()

    def get_conn(self):
        conn = self.conn_pool.getconn()
        conn.autocommit = True
        return conn

    def put_conn(self, conn):
        return self.conn_pool.putconn(conn)


@contextmanager
def get_db_connection():
    """Context manager providing a connection from the connection pool.

    Ensures the connection is put back to the pool when finished and handles errors.

    Usage:

    with pg_db.get_db_connection() as conn:
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
            # assume that we need to reinitialize connection pool
            # when any pg_db error happens
            pool.reset()
            raise db_error
        pool.put_conn(conn)


def run_query(query: str, params: Dict = None) -> pd.DataFrame:
    """Gets a connection from the pool and executes the query
    with the specified parameters.

    Returns a DataFrame with the results.
    """
    with get_db_connection() as conn:
        return pd.read_sql_query(query, conn, params=params)


pool = ConnectionPool()
