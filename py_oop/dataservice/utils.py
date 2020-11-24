import json
import logging
import os
import re
from datetime import datetime, timezone
from functools import wraps
from io import StringIO
from typing import Union

import pandas as pd
import psycopg2
from pandas.errors import ParserError
from pythonjsonlogger.jsonlogger import JsonFormatter

from py_oop.dataservice import config

logger = logging.getLogger(__name__)


def audited(f):
    """
    Decorate a function to time it and log it out (also captures args/kwargs)
    """

    @wraps(f)
    def wrap(*args, **kw):
        logger.debug(f"START: func:{f.__name__}", extra={"func_args": args, "func_kwargs": kw})

        start = datetime.now()
        result = f(*args, **kw)
        end = datetime.now()

        logger.debug(
            f"END: func:{f.__name__}", extra={"func_args": args, "func_kwargs": kw, "time": f"{end - start}"},
        )
        return result

    return wrap


def timed(f):
    """
    Decorate a function to time it and log it out
    """

    @wraps(f)
    def wrap(*args, **kw):
        start = datetime.now()
        result = f(*args, **kw)
        end = datetime.now()
        logger.debug(f"func:{f.__name__!r} took: {end - start} sec")
        return result

    return wrap


class CustomJsonFormatter(JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        """
        See https://cloud.google.com/logging/docs/agent/configuration#special-fields for all special fields we can use
        See https://cloud.google.com/run/docs/logging for logging setup
        """
        super().add_fields(log_record, record, message_dict)
        local_time = datetime.now(timezone.utc).astimezone()
        now = local_time.isoformat()

        action = log_record.pop("message")
        message = log_record.copy()
        for key in list(log_record.keys()):
            del log_record[key]

        message["action"] = action
        log_record["message"] = message
        log_record["time"] = now
        log_record["severity"] = record.levelname


def get_conn():
    conn = psycopg2.connect(
        host=config.POSTGRES_HOST,
        port=config.POSTGRES_PORT,
        dbname=config.POSTGRES_DB,
        user=config.POSTGRES_USER,
        password=config.POSTGRES_PASS,
    )
    return conn


@timed
def read_sql_inmem_uncompressed(query: str) -> pd.DataFrame:
    """
    Grab data out of the db in the most efficient way possible
    """
    copy_sql = f"COPY ({query}) TO STDOUT WITH CSV HEADER"
    cursor = get_conn().cursor()
    store = StringIO()
    cursor.copy_expert(copy_sql, store)
    store.seek(0)

    try:
        df = pd.read_csv(store, encoding="utf8", low_memory=False)
    except ParserError as exc:
        logger.error(exc)
        raise

    return df


def check_and_sort_cols(df: pd.DataFrame, table: str, cursor, schema: str = "public") -> Union[None, pd.DataFrame]:
    """
    Ensure we have all the columns required, and they are in the right order
    """
    cursor.execute(f"SELECT * FROM {schema}.{table} LIMIT 0;")
    db_cols = [x.name for x in cursor.description]
    df_cols = df.columns.tolist()

    try:
        df = df[db_cols]
    except KeyError as exc:
        logger.error(
            f"Column Mismatch on {table} ({exc})", extra={"db_cols": db_cols, "df_cols": df_cols, "table": table},
        )
        return None

    return df


def underscore_to_camel(name: str) -> str:
    """
    Convert a name from underscore lower case convention to camel case convention.
    Args:
        name (str): name in underscore lowercase convention.
    Returns:
        Name in camel case convention.
    """
    under_pat = re.compile(r"_([a-z])")
    return under_pat.sub(lambda x: x.group(1).upper(), name)


def get_current_db_id() -> int:
    filename = "/tmp/database.json"
    if config.DEBUG or not os.path.isfile(filename):
        return 0

    with open(filename) as json_file:
        data = json.load(json_file)

    return data.get("db_id")


def get_previous_week_no() -> int:
    """
    Get the epoch-like week number for last week (This is how we deal with time/date querying)
    """
    return int(int(datetime.now().timestamp()) / 60 / 60 / 24 / 7) - 1
