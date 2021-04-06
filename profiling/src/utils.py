import logging
import re
import time
from functools import wraps
from time import perf_counter

logger = logging.getLogger(__name__)


def time_fn(wrapped_func):
    """Decorate a function to time it and log it out"""

    @wraps(wrapped_func)
    def measure_time(*args, **kw):
        start = time.time()
        result = wrapped_func(*args, **kw)
        end = time.time()
        print(f"@timefn: {wrapped_func.__name__} took {end - start} seconds")
        return result

    return measure_time


def timed(wrapped_func):
    """Decorate a function to time it and log it out"""

    @wraps(wrapped_func)
    def measure_time(*args, **kw):
        start = perf_counter()
        result = wrapped_func(*args, **kw)
        end = perf_counter()
        logger.debug("func: %s took: %2.4f sec", wrapped_func.__name__, end - start)
        return result

    return measure_time


def audited(wrapped_func):
    """Decorate a function to time it and log it out (also captures args/kwargs)"""

    @wraps(wrapped_func)
    def audit(*args, **kw):
        logger.debug(
            "[START] func: %s",
            wrapped_func.__name__,
            extra={"func_args": args, "func_kwargs": kw},
        )

        start = perf_counter()
        result = wrapped_func(*args, **kw)
        end = perf_counter()

        logger.debug(
            "END: func: %s",
            wrapped_func.__name__,
            extra={"func_args": args, "func_kwargs": kw, "time": f"{end - start:2.4f}"},
        )
        return result

    return audit


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
