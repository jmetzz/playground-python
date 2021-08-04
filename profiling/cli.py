"""This is the driver command line application
to collect profiling information (CPU and memory)
about the ml module.
"""
import gc
import logging.config
import pstats
import resource
import shutil
import subprocess
import timeit
from cProfile import Profile
from pathlib import Path
from string import Template
from tempfile import NamedTemporaryFile
from time import sleep

import numpy as np

import click
import psutil
import pyinstrument
import pyinstrument_flame

from cli_config import (
    THIS_DIR,
    LOGGING_CONFIG,
)
from algos.juliaset import calc_pure_python

logging.config.dictConfig(LOGGING_CONFIG)

logger = logging.getLogger(__name__)

SETUP_TEMPLATE = """
"""

STMT_TEMPLATE = """
"""


@click.group()
def main():
    """Just a simple click group to manage commands."""


# pylint: disable="too-many-arguments"
@main.command()
@click.option("--width", type=int, default=1000)
@click.option("--iterations", type=int, default=300)
@click.option("--pretty", is_flag=True)
@click.option("--save", is_flag=True)
@click.option("--vis", is_flag=True)
@click.option("--flame", is_flag=True)
def cpu(width, iterations, pretty, save, vis, flame) -> None:
    """Wrapper to trigger cpu profiling"""
    if pretty:
        use_pyinstrument(width, iterations, save, flame)
    else:
        use_cprofiler(width, iterations, save, vis)


def use_cprofiler(
    width: int,
    iterations: int,
    save: bool = False,
    visualize: bool = False,
) -> None:
    """Run cProfiler to measure CPU performance"""
    with Profile() as c_profiler:
        calc_pure_python(desired_width=width, max_iterations=iterations)

    with NamedTemporaryFile(delete=False) as tmp_file:
        filename = tmp_file.name
        if save:
            Path(THIS_DIR, ".profile_data").mkdir(parents=True, exist_ok=True)
            c_profiler.dump_stats(
                str(Path(THIS_DIR, ".profile_data", f"{width}-{iterations}-cprof.prof"))
            )
            print(
                f"Output file written: '{THIS_DIR}/.profile_data/{width}-{iterations}-cprof.prof'"
            )
        if visualize:
            c_profiler.dump_stats(filename)
            subprocess.Popen(["snakeviz", filename])
        else:
            # sort by time
            c_profiler.print_stats(sort=1)


def use_pyinstrument(
    width: int,
    iterations: int,
    save: bool = False,
    flame: bool = False,
) -> None:
    """Run pyinstrument to measure CPU performance"""

    inst_profiler = pyinstrument.Profiler()
    inst_profiler.start()
    calc_pure_python(desired_width=width, max_iterations=iterations)
    inst_profiler.stop()

    print(inst_profiler.output_text(unicode=True, color=True))

    if save:
        Path(THIS_DIR, ".profile_data").mkdir(parents=True, exist_ok=True)
        filename = str(Path(THIS_DIR, ".profile_data", f"{width}-{iterations}"))
        with open(f"{filename}-pyinst.html", "w") as file:
            file.write(inst_profiler.output_html())

        if flame:
            renderer = pyinstrument_flame.FlameGraphRenderer(
                title=f"{width}-{iterations}", flamechart=True
            )
            with open(f"{width}-{iterations}-pyinst-flame.html", "w") as file:
                file.write(inst_profiler.output(renderer))


# pylint: disable="no-member"
@main.command()
@click.option("--file", type=str)
@click.option("--limit", type=click.INT, default=None)
@click.option("--callers", is_flag=True)
@click.option(
    "--sortby",
    type=click.Choice(pstats.SortKey.__members__.keys(), case_sensitive=False),
    default="CUMULATIVE",
)
def stats(file, limit, callers, sortby) -> None:
    """Command to read cProfiler output"""
    stats_obj = pstats.Stats(file)

    key = pstats.SortKey(sortby.lower())
    stats_obj.strip_dirs().sort_stats(key).print_stats(limit)
    if callers:
        stats_obj.print_callees(limit)


@main.command()
@click.option("--width", type=int, default=1000)
@click.option("--iterations", type=int, default=300)
@click.option("--save", is_flag=True)
def mem(width: int, iterations: int, save: bool) -> None:
    """Checks the memory usage

    This function is equivalent to run the following commands from the prompt:

    ```bash
    python -m mprof run -o <output-file> <target-python-script> <script params>
    python -m mprof plot -t "Memory usage over time" <output-file> --backend MacOSX
    ```
    """

    from memory_profiler import memory_usage

    with NamedTemporaryFile(delete=False, mode="w+") as tmp_file:
        usage = memory_usage((calc_pure_python, (width, iterations)), timestamps=True)
        tmp_file.write("CMDLINE python cli.py mem-train\n")
        content = [f"MEM {mem:.6f} {ts:.4f}\n" for mem, ts in usage]
        tmp_file.writelines(content)

    subprocess.run(
        ["python", "-m", "mprof", "plot", "--flame", tmp_file.name], check=True
    )

    if save:
        Path(THIS_DIR, ".profile_data").mkdir(parents=True, exist_ok=True)
        filename = str(Path(THIS_DIR, ".profile_data", f"{width}-{iterations}.dot"))
        shutil.move(tmp_file.name, filename)
    print("You can plot the chart again with:")
    print(
        "\t$ python -m mprof plot -t 'Your title' <width-iterations.dot> --backend MacOSX"
    )


@main.command()
@click.option("--rcm-dir", type=str, default="", help="Path the RCM models directory")
@click.option(
    "--batch",
    is_flag=True,
    help="Triggers the execution of MM 20 times, increasing the size of the input data by" " 5% at each iteration.",
)
@click.option(
    "--fast",
    is_flag=True,
    help="Triggers the execution using a known small cell (TW, WASHING_MACHINES, period 2677),"
    " When not set, the big cell (FR, HOT_BEV, 2655) is used.",
)
def trace_mem(rcm_dir, batch, fast):

    if fast:
        # small data config
        width = 5
        iterations = 1
    else:
        # big data config
        width = 10
        iterations = 100

    schedule = np.linspace(1.0, 2.0, 20) if batch else [1.]

    summary_info = {}
    for it, sampling_ratio in enumerate(schedule):
        logger.debug("===================== STARTING TRACE MEMORY USAGE PROFILING =====================")
        logger.debug("[Start] Iteration '%d'", it)
        logger.debug("Oversampling percentage '%.2f'", sampling_ratio)
        # psutil express values in bytes.
        # see https://psutil.readthedocs.io/en/latest/#psutil.Process.memory_info
        psutil_memory_info = psutil.Process().memory_info()
        rss_start, vms_start = (psutil_memory_info.rss / 1024 ** 2, psutil_memory_info.vms / 1024 ** 2)

        calc_pure_python_with_tracemalloc_wrapper(desired_width=width, max_iterations=iterations)
        psutil_memory_info = psutil.Process().memory_info()
        rss_end, vms_end = (psutil_memory_info.rss / 1024 ** 2, psutil_memory_info.vms / 1024 ** 2)

        # resource ru_maxrss is expressed in kilobytes.
        # see https://docs.python.org/3/library/resource.html#resource.getrusage
        #     ru_maxrss: maximum resident set size
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        summary_info[sampling_ratio] = {
            "init_rss": rss_start,
            "final_rss": rss_end,
            "init_vms": vms_start,
            "final_vms": vms_end,
            "peak_rss": peak_rss
        }
        logger.debug("Initial Resident Set Size & Virtual Memory Size  '(%d, %d)' MB", rss_start, vms_start)
        logger.debug("Final Resident Set Size & Virtual Memory Size '(%d, %d)' MB", rss_end, vms_end)
        logger.debug("Peak Resident Set Size '%d' MB", peak_rss)
        logger.debug("[End] Iteration '%d' ------------------------------------------------------------", it)
        gc.collect()
        # give some time to the Garbage collector do it's thing
        sleep(5)

    print(summary_info)


@main.command()
@click.option("--loops", type=click.INT, default=None)
@click.option("--reps", type=click.INT, default=1)
def timeit_cli(loops, reps) -> None:
    """Utility function to calculate the time to run a code statement"""

    setup_code = Template(SETUP_TEMPLATE).substitute()

    stmt_code = Template(STMT_TEMPLATE).substitute()
    if reps and loops:
        outcome = timeit.repeat(
            setup=setup_code, stmt=stmt_code, repeat=reps, number=loops
        )
        print(
            f"Best time '{min(outcome)}'s; '{loops}' loops and {reps} repeats per loop"
        )
    else:
        loops, outcome = timeit.Timer(setup=setup_code, stmt=stmt_code).autorange()
        print(f"Execution time '{outcome}'s in '{loops}' loops")


if __name__ == "__main__":
    main()
