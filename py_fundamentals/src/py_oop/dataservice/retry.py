import functools
import logging
from time import sleep

logger = logging.getLogger(__name__)


def retry(max_tries=3, delay=0, exceptions=()):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            exception = None
            for attempt in range(max_tries):
                try:
                    return f(*args, **kwargs)
                except exceptions as e:
                    exception = e
                    if attempt + 1 < max_tries:
                        logger.warning(
                            "(%d/%d) waiting for %s seconds before retrying again",
                            attempt + 1,
                            max_tries,
                            delay,
                        )
                        sleep(delay)
                    if attempt + 1 == max_tries:
                        logger.error(
                            "(%d/%d) exceeded max tries [%s]",
                            attempt + 1,
                            max_tries,
                            exception,
                        )
                    pass
            raise exception

        return wrapper

    return decorator
