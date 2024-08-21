"""This is the driver command line application
to collect profiling information (CPU and memory)
about the ml module.
"""

import logging.config
import pstats
import shutil
import subprocess
import timeit
from cProfile import Profile
from pathlib import Path
from string import Template
from tempfile import NamedTemporaryFile
from typing import Optional

import pyinstrument
import pyinstrument_flame
import typer
from cli_config import LOGGING_CONFIG, THIS_DIR

from algorithms.juliaset import calc_pure_python

app = typer.Typer()

logging.config.dictConfig(LOGGING_CONFIG)

logger = logging.getLogger(__name__)

SETUP_TEMPLATE = """
"""

STMT_TEMPLATE = """
"""

PROFILE_DATA_FILE = ".profile_data"


@app.command()
def cpu(
    width: int = typer.Option(1000),
    iterations: int = typer.Option(300),
    pretty: bool = typer.Option(False),
    save: bool = typer.Option(False),
    vis: bool = typer.Option(False),
    flame: bool = typer.Option(False),
) -> None:
    """Wrapper to trigger cpu profiling"""
    if pretty:
        use_pyinstrument(width, iterations, save, flame)
    else:
        use_cprofiler(width, iterations, save, vis)


@app.command()
def stats(
    file: str,
    limit: int = typer.Option(None),
    callers: bool = typer.Option(False),
    sortby: pstats.SortKey = pstats.SortKey.CUMULATIVE,
) -> None:
    """Command to read cProfiler output"""
    stats_obj = pstats.Stats(file)

    key = sortby
    stats_obj.strip_dirs().sort_stats(key).print_stats(limit)
    if callers:
        stats_obj.print_callees(limit)


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
            Path(THIS_DIR, PROFILE_DATA_FILE).mkdir(parents=True, exist_ok=True)
            c_profiler.dump_stats(str(Path(THIS_DIR, PROFILE_DATA_FILE, f"{width}-{iterations}-cprof.prof")))
            print(f"Output file written: '{THIS_DIR}/{PROFILE_DATA_FILE}/{width}-{iterations}-cprof.prof'")
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
        Path(THIS_DIR, PROFILE_DATA_FILE).mkdir(parents=True, exist_ok=True)
        filename = str(Path(THIS_DIR, PROFILE_DATA_FILE, f"{width}-{iterations}"))
        with open(f"{filename}-pyinst.html", "w") as file:
            file.write(inst_profiler.output_html())

        if flame:
            renderer = pyinstrument_flame.FlameGraphRenderer(title=f"{width}-{iterations}", flamechart=True)
            with open(f"{width}-{iterations}-pyinst-flame.html", "w") as file:
                file.write(inst_profiler.output(renderer))


# pylint: disable="no-member"


@app.command()
def mem(
    width: int = typer.Option(1000, "--width"),
    iterations: int = typer.Option(300, "--iterations"),
    save: bool = typer.Option(False, "--save"),
) -> None:
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

    subprocess.run(["python", "-m", "mprof", "plot", "--flame", tmp_file.name], check=True)

    if save:
        Path(THIS_DIR, PROFILE_DATA_FILE).mkdir(parents=True, exist_ok=True)
        filename = str(Path(THIS_DIR, PROFILE_DATA_FILE, f"{width}-{iterations}.dot"))
        shutil.move(tmp_file.name, filename)
    print("You can plot the chart again with:")
    print("\t$ python -m mprof plot -t 'Your title' <width-iterations.dot> --backend MacOSX")


@app.command()
def timeit_cli(loops: Optional[int] = typer.Option(None, "--loops"), reps: int = typer.Option(1, "--reps")) -> None:
    """Utility function to calculate the time to run a code statement"""
    setup_code = Template(SETUP_TEMPLATE).substitute()

    stmt_code = Template(STMT_TEMPLATE).substitute()
    if reps and loops:
        outcome = timeit.repeat(setup=setup_code, stmt=stmt_code, repeat=reps, number=loops)
        print(f"Best time '{min(outcome)}'s; '{loops}' loops and {reps} repeats per loop")
    else:
        loops, outcome = timeit.Timer(setup=setup_code, stmt=stmt_code).autorange()
        print(f"Execution time '{outcome}'s in '{loops}' loops")


if __name__ == "__main__":
    app()
