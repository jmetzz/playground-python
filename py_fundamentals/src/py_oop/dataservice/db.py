import logging
from contextlib import contextmanager
from tempfile import TemporaryFile
from threading import Semaphore
from typing import Any, Dict, Optional

import pandas as pd
from psycopg2 import (
    DatabaseError,
    InterfaceError,
    OperationalError,
    ProgrammingError,
    pool,
)
from psycopg2.extensions import connection

logger = logging.getLogger(__name__)

EXCEPTIONS = (
    InterfaceError,
    OperationalError,
    DatabaseError,
    ProgrammingError,
    pd.io.sql.DatabaseError,
)


class BlockingThreadedConnectionPool(pool.ThreadedConnectionPool):
    """Blocks on getconn to wait for a connection to become free."""

    def __init__(self, minconn, maxconn, *args, **kwargs):
        self._semaphore = Semaphore(maxconn)
        super().__init__(minconn, maxconn, *args, **kwargs)

    def getconn(self, *args, **kwargs):
        self._semaphore.acquire()
        try:
            return super().getconn(*args, **kwargs)
        except Exception as e:
            self._semaphore.release()
            raise e

    def putconn(self, *args, **kwargs):
        super().putconn(*args, **kwargs)
        self._semaphore.release()


def make_conn_pool(
    postgres_host: str,
    postgres_port: str,
    postgres_db: str,
    postgres_user: str,
    postgres_pass: str,
    connect_timeout: int = 3,
    max_conn_count: int = 10,
    min_conn_count: int = 1,
    application_name: str = "data_access",
) -> pool.AbstractConnectionPool:
    """Creates a connection pool."""
    return BlockingThreadedConnectionPool(
        minconn=min_conn_count,
        maxconn=max_conn_count,
        host=postgres_host,
        port=postgres_port,
        dbname=postgres_db,
        user=postgres_user,
        password=postgres_pass,
        connect_timeout=connect_timeout,
        application_name=application_name,
    )


@contextmanager
def get_conn_pool(
    postgres_host: str,
    postgres_port: str,
    postgres_db: str,
    postgres_user: str,
    postgres_pass: str,
    connect_timeout: int = 3,
    max_conn_count: int = 10,
    min_conn_count: int = 1,
):
    """Context manager that sets up a connection pool, the pool will close when
    exiting the context.

    Usage:

    with get_conn_pool(config) as pool:
        ...
    """
    connection_pool = make_conn_pool(
        postgres_host,
        postgres_port,
        postgres_db,
        postgres_user,
        postgres_pass,
        connect_timeout,
        max_conn_count,
        min_conn_count,
    )
    if connection_pool is None:
        raise RuntimeError("Could not create the connection pool")
    try:
        yield connection_pool
    finally:
        connection_pool.closeall()


@contextmanager
def get_db_connection(connection_pool: pool.AbstractConnectionPool):
    """Context manager that returns a connection from the supplied pool.

    The connection will be put back to the pool when exiting the context and
    will be closed in case of an error.

    Args:
        connection_pool (AbstractConnectionPool): Connection pool to pull the
            connection from.

    Usage:

    with get_db_connection(my_pool) as conn:
        ...
    """
    if connection_pool is None:
        raise RuntimeError("Connection pool is None")
    db_error = None
    conn = None
    try:
        conn = connection_pool.getconn()
        conn.autocommit = True
        yield conn
    except EXCEPTIONS as e:
        db_error = e
    finally:
        if conn is not None:
            connection_pool.putconn(conn, close=(db_error is not None))
        if db_error is not None:
            purge_unhealthy_connections(connection_pool)
            raise db_error


def purge_unhealthy_connections(connection_pool: pool.AbstractConnectionPool):
    """Discards unhealthy connections from the pool.

    psycopg2's connection pool cannot detect when the underlying connections
    are broken, e.g. when the database restarts. This function pulls all the
    connections in the pool until a healthy one is returned which may be a
    newly created one in the worst case.
    """
    logger.info("Purging connection pool")
    conn = connection_pool.getconn()
    while not is_connection_healthy(conn):
        connection_pool.putconn(conn, close=True)
        conn = connection_pool.getconn()

    # put healthy connection back in the pool
    connection_pool.putconn(conn, close=False)


def is_connection_healthy(conn: connection):
    try:
        with conn.cursor() as cursor:
            cursor.execute("select 1")
            cursor.fetchone()
    except Exception:
        return False
    else:
        return True


default_connection_pool = None


class DataAccess:
    """
    Class that encapsulates database access.

    Args:
        connection_pool (AbstractConnectionPool): Optional connection pool to
            be used. If none is supplied a new one will be created and set
            as the default to be reused on subsequent instantiations.
    """

    configuration: dict = None

    def __init__(self, connection_pool: pool.AbstractConnectionPool = None):
        global default_connection_pool
        if connection_pool is None:
            if default_connection_pool is None:
                logger.info("setting default connection pool")
                if self.configuration is None:
                    raise ValueError(
                        "No configuration set, please call DataAccess.configure"
                    )
                connection_pool = default_connection_pool = make_conn_pool(
                    **self.configuration
                )
            else:
                connection_pool = default_connection_pool

        self.connection_pool = connection_pool

    @classmethod
    def configure(
        cls,
        postgres_host: str,
        postgres_port: str,
        postgres_db: str,
        postgres_user: str,
        postgres_pass: str,
        connect_timeout: int = 3,
        max_conn_count: int = 10,
        min_conn_count: int = 1,
        application_name: str = "data_access",
    ):
        """Set the configuration for the default connection pool."""
        cls.configuration = {
            "postgres_host": postgres_host,
            "postgres_port": postgres_port,
            "postgres_db": postgres_db,
            "postgres_user": postgres_user,
            "postgres_pass": postgres_pass,
            "connect_timeout": connect_timeout,
            "max_conn_count": max_conn_count,
            "min_conn_count": min_conn_count,
            "application_name": application_name,
        }

    def recreate_pool(self):
        """Close the connection pool and create a new one based on the configuration."""
        self.connection_pool.closeall()
        self.connection_pool = make_conn_pool(**self.configuration)

    def _execute_sql_query(
        self, sql_query: str, params: Dict[str, Any], return_all: Optional[bool] = None
    ):
        with get_db_connection(self.connection_pool) as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql_query, params or {})
                if return_all is not None:
                    if return_all:
                        return cursor.fetchall()
                    return cursor.fetchone()
        return None

    def execute_sql_query(self, sql_query: str, params=None):
        """Execute an SQL query optionally with parameters.

        TODO: this is misleading as the only valid queries are SELECT.
        """
        return self.fetch_all(sql_query, params)

    def fetch_all(self, sql_query, params=None):
        """Execute an SQL query and return all the returned rows."""
        return self._execute_sql_query(sql_query, params, True)

    def fetch_one(self, sql_query, params=None):
        """Execute an SQL query and return a single row."""
        return self._execute_sql_query(sql_query, params, False)

    def execute(self, sql_query, params=None):
        """Execute a single query with the parameters, does not fetch results."""
        self._execute_sql_query(sql_query, params)

    def update(self, sql_query, params=None):
        # TODO: this should be named more generically or force and UPDATE query
        self.execute(sql_query, params)

    def read_sql_query(self, sql_query, params=None):
        """
        Run a query with optional parameters and recursive trying to overcome
        invalid connections.

        Args:
            sql_query (str): sql query with optional parameters, interface
                format pyformat %()s.
            params (list): Optional parameters to insert in query.

        Returns:
            pd.DataFrame: query output.
        """
        with get_db_connection(self.connection_pool) as conn:
            return pd.read_sql_query(sql=sql_query, con=conn, params=params)

    def read_psql_with_copy_command(self, sql_query, params=None, *args, **kwargs):
        """
        Read and build a DF using the copy_expert psycopg2 command.

        Better suited for retrieving large data sets.

        This command will only work with PostgreSQL DBs. The 'COPY' command of
        postgres is proven to be the fastest way of extracting data from the DB,
        especially when it comes to mass retrievals.

        This function takes advantage of the 'COPY' command, uses it to create
        a temporary CSV file in the memory, and makes pandas read_csv from that
        temporary file.

        Args:
            sql_query (str): The PostgreSQL query to be executed
            params (dict): Optional parameters of the query
            args: Any additional arguments to be passed on to pd.read_csv.
            kwargs: Any additional keyword arguments to be passed on to
                pd.read_csv. e.g. dtype = {...}

        Returns:
            pd.DataFrame: Output of the query.
        """
        if params:
            sql_query = sql_query % params

        with get_db_connection(self.connection_pool) as conn:
            with conn.cursor() as cursor:
                # TODO: consider using StringIO objects to avoid writing to disk
                with TemporaryFile() as temp_file:
                    copy_sql = "COPY ({query}) TO STDOUT WITH CSV HEADER".format(
                        query=sql_query
                    )
                    cursor.copy_expert(copy_sql, temp_file)
                    temp_file.seek(0)
                    df = pd.read_csv(
                        temp_file, *args, **kwargs, float_precision="round_trip"
                    )
        return df
