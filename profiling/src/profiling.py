"""
Profiling wrappers and helpers.
"""
import linecache
import logging
import os
import tracemalloc
from datetime import datetime
from functools import wraps
from time import perf_counter
from typing import Iterable, List, Optional

import psutil
from environs import Env

logger = logging.getLogger("MODELMAKER_PROFILER")
logger.setLevel(logging.DEBUG)

env = Env()
env.read_env()

TRACEMALLOC_ENABLED = env.bool("TRACEMALLOC_ENABLED", os.getenv("TRACEMALLOC_ENABLED") == 1)
MB_MODIFIER = 1024 ** 2


def timed(log_level=logging.DEBUG):
    """
    Decorate a function to time it and log it out

    Usage:
        decorate the target function with `@timed()`.
        You can optionally specify with log level to use.

        NOTE though, the parenthesis are required!
        do not use only `@timed`.
    """

    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = datetime.now()
            result = func(*args, **kwargs)
            end = datetime.now()
            logger.log(log_level, "ELAPSED TIME: func:%s in '%s'", func.__name__, end - start)
            return result

        return wrapper

    return inner


def timed_counter(log_level=logging.DEBUG):
    """
    Decorate a function to time it and log it using time.perf_counter.

    Logs the value (in fractional seconds) of a performance counter.
    It does include time elapsed during sleep and is system-wide.
    The reference point of the returned value is undefined,
    so that only the difference between the results of two calls is valid.

    Usage:
       decorate the target function with `@timed()`.
       You can optionally specify with log level to use.

       NOTE though, the parenthesis are required!
       do not use only `@timed`.
    """

    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = perf_counter()
            result = func(*args, **kwargs)
            end = perf_counter()
            logger.log(
                log_level,
                "PERF COUNTER: %s took: %2.4f sec",
                func.__name__,
                end - start,
            )
            return result

        return wrapper

    return inner


def mem_audited(log_level=logging.DEBUG):
    """
    Decorate a function to log memory usage information

    Usage:
        decorate the target function with `@mem_audited()`.
        You can optionally specify with log level to use.

        NOTE though, the parenthesis are required!
        do not use only `@mem_audited`.
    """

    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # psutil express values in bytes.
            # see https://psutil.readthedocs.io/en/latest/#psutil.Process.memory_info

            psutil_memory_info = psutil.Process().memory_info()
            rss_start, vms_start = (
                psutil_memory_info.rss / MB_MODIFIER,
                psutil_memory_info.vms / MB_MODIFIER,
            )
            result = func(*args, **kwargs)
            psutil_memory_info = psutil.Process().memory_info()
            rss_end, vms_end = (
                psutil_memory_info.rss / MB_MODIFIER,
                psutil_memory_info.vms / MB_MODIFIER,
            )
            logger.log(
                log_level,
                "PROCESS MEMORY USAGE: func:%s [before exec '(%.2f, %.2f) MB', "
                "after exec '(%.2f, %.2f) MB', delta '(%.2f, %.2f) MB']",
                func.__name__,
                rss_start,
                vms_start,
                rss_end,
                vms_end,
                rss_end - rss_start,
                vms_end - vms_start,
            )
            return result

        return wrapper

    return inner


def audit(func):
    """
    Decorate a function to time it and log it out (also captures args/kwargs)
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(
            "[START] func: %s",
            func.__name__,
            extra={"func_args": args, "func_kwargs": kwargs},
        )

        start = perf_counter()
        result = func(*args, **kwargs)
        end = perf_counter()

        logger.debug(
            "END: func: %s",
            func.__name__,
            extra={
                "func_args": args,
                "func_kwargs": kwargs,
                "time": f"{end - start:2.4f}",
            },
        )
        return result

    return wrapper


class TracemallocWrapper:
    # pylint: disable=too-few-public-methods
    """Profiles memory allocation"""

    @staticmethod
    def take_snapshot_and_log_stats(filters: Optional[Iterable] = None, msg: str = ""):
        """
        Takes tracemalloc snapshot and prints statistics
        :param filters: Filters to include/exclude python modules.
        :param msg: Extra massage/info to print.
        """
        if not TRACEMALLOC_ENABLED:
            return

        if not logger.isEnabledFor(logging.DEBUG):
            return

        logger.debug("[START OF MEMORY TRACKING BLOCK]")
        curr_snapshot = TracemallocWrapper.take_snapshot(filters)
        TracemallocWrapper._display_snapshot_stats(curr_snapshot)
        logger.debug("Biggest memory block:")
        TracemallocWrapper._display_snapshot_stats(
            curr_snapshot,
            key_type="traceback",
            limit=1,
            show_line=True,
            show_traceback=True,
        )
        logger.debug("[END OF MEMORY TRACKING BLOCK]")
        return curr_snapshot

    @staticmethod
    def take_snapshot(
        filters: Optional[Iterable],
    ):
        """
        Takes snapshot.
        :param filters: Applies provided filters.
        :return: Current snapshot.
        """
        if not TRACEMALLOC_ENABLED:
            return
        curr_snapshot = tracemalloc.take_snapshot()
        curr_snapshot = TracemallocWrapper._filter_traces(curr_snapshot, filters)
        return curr_snapshot

    @staticmethod
    def log_snapshots_diff(
        snapshot_1: tracemalloc.Snapshot,
        snapshot_2: tracemalloc.Snapshot,
        key_type="lineno",
        limit=10,
        show_line=False,
        show_traceback=False,
    ):
        # pylint: disable=too-many-arguments
        """
        Prints difference between two snapshots.
        :param snapshot_1: First snapshot
        :param snapshot_2: Second snapshot
        :param key_type: Statistics key
        :param limit: Limit number
        :param show_line: True - show first line, false - otherwise
        :param show_traceback: True - show traceback frames, false - otherwise
        """
        if not logger.isEnabledFor(logging.DEBUG):
            return
        logger.debug("[START OF MEMORY SNAPSHOT COMPARISON BLOCK]")
        top_stats = snapshot_2.compare_to(snapshot_1, key_type)
        TracemallocWrapper._display_top_stats(top_stats, limit, show_line, show_traceback)
        logger.debug("[END OF MEMORY SNAPSHOT COMPARISON BLOCK]")

    @staticmethod
    def log_object_traceback(obj):
        """
        Prints object traceback.
        :param obj: Object to trace.
        """
        obj_traceback = tracemalloc.get_object_traceback(obj)
        TracemallocWrapper.log_traceback(obj_traceback)

    @staticmethod
    def _display_snapshot_stats(snapshot, key_type="lineno", limit=10, show_line=False, show_traceback=False):
        top_stats = snapshot.statistics(key_type)
        TracemallocWrapper._display_top_stats(top_stats, limit, show_line, show_traceback)

    @staticmethod
    def _display_top_stats(
        top_stats: List[tracemalloc.Statistic],
        limit=10,
        show_line=False,
        show_traceback=False,
    ):
        if not logger.isEnabledFor(logging.DEBUG):
            return
        if not top_stats:
            return
        logger.debug("Top %s lines", limit)
        for index, stat in enumerate(top_stats[:limit], 0):
            logger.debug("#%s: %s", index + 1, stat)
            curr_traceback = stat.traceback
            if show_line:
                frame = curr_traceback[0]
                line = linecache.getline(frame.filename, frame.lineno).strip()
                if line:
                    logger.debug("    %s", line)
            if show_traceback:
                TracemallocWrapper.log_traceback(curr_traceback)
        other = top_stats[limit:]
        if other:
            size = sum(stat.size for stat in other)
            logger.debug(
                "%s other lines: cummulative size=%s",
                len(other),
                tracemalloc._format_size(size, False),  # pylint: disable=protected-access
            )
        total = sum(stat.size for stat in top_stats)
        logger.debug(
            "Total allocated size: %s",
            tracemalloc._format_size(total, False),  # pylint: disable=protected-access
        )

    @staticmethod
    def log_traceback(
        traceback: Optional[tracemalloc.Traceback],
    ):
        """
        Print traceback frames.
        :param traceback: Traceback instance
        :return: Nothing
        """
        if not logger.isEnabledFor(logging.DEBUG):
            return
        if not traceback:
            logger.warning("Provided traceback is None")
            return
        index = 0
        logger.debug("Traceback (most recent call first):")
        for line in traceback:
            logger.debug("#%s: %s", index + 1, line)
            index = index + 1

    @staticmethod
    def _filter_traces(snapshot, filters: Iterable):
        predefined_filters = (
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<frozen importlib._bootstrap_external>"),
            tracemalloc.Filter(False, "<unknown>"),
            tracemalloc.Filter(False, tracemalloc.__file__),
        )
        snapshot = snapshot.filter_traces(predefined_filters)
        if filters:
            snapshot = snapshot.filter_traces(filters)
        return snapshot

    @staticmethod
    def start(nframe: int = 1):
        """
        Start memory profiling.
        """
        if not TRACEMALLOC_ENABLED:
            return
        tracemalloc.start(nframe)

    @staticmethod
    def stop():
        """
        Start memory profiling.
        """
        tracemalloc.stop()

    @staticmethod
    def log_summary():
        """
        tracemalloc memory overhead in kilobytes
        to store traces of memory blocks.
        """
        if not logger.isEnabledFor(logging.DEBUG):
            return
        if not tracemalloc.is_tracing():
            return
        tracer_mem = tracemalloc.get_tracemalloc_memory()
        current, peak = tracemalloc.get_traced_memory()
        logger.debug(f"Current memory usage is {current / 1024 ** 2} MB")
        logger.debug(f"Peak memory usage was {peak / 1024 ** 2} MB")
        logger.debug("tracemalloc memory overhead: %d MB", tracer_mem / (1024 ** 2))
