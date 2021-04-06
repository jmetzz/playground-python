# Profiling python code

> Find bottlenecks so we can do the least amount of work to ge the biggest practical performance gain [Gorelick & Ozvald, 2020]


How do you tell if changes you have made to the code have actually improved the performance? You must observe the normal variation when you’re timing your code, or you might incorrectly attribute an improvement in your code to what is simply a random variation in execution time. This means, the computer performs many other tasks while running your code, such as accessing the network, disk, or RAM, and these factors can cause variations in the execution time of your program. Moreover, **profiling typically adds an overhead** (`10x` to `100x` slowdowns can be typical). So you cannot use the timings you get from profiling as absolution measures of how your code will run in production. Actually, without profiling your code you will run much faster.



Once you the requirements covered (hypotesis, realistic scenario, realistic input data), then you can start thinking about how to measure the perfromance. There are mainly 2 types of profiling tools you should be aware of:

- **Deterministic** (or event-based) profiling is meant to reflect the fact that all function call, function return, and exception events are monitored, and precise timings are made for the intervals between these events.
- **Statistical** profiling randomly samples the effective instruction pointer, and deduces where time is being spent. This approach provides only relative indications of where time is being spent.

For more details [see the python official documentation.](https://docs.python.org/3/library/profile.html).



## Setup


In order to collect profiling information from a python application, you first need to install the supporting python modules that will collect `cpu` and `memory` usage from the target running process.

This project uses `poetry` as a package/dependencies management tool. Moreover, I've included some tasks via `Makefile` to simplify the environment creation, dependencies update, code quality checks, linting, and cleaning the workspace.

Setup your python dependencies with:

- `make interpreter` installs `pyenv`, the required python version, and `poetry`
- `make deps` installs all the python dependencies declared in the `pyproject.toml` file.


I have already configured Jupyter lab as extra dependencies. If you need it just run `poetry install --extras jupyter`. 

You can add other dependencies as you go by running

```bash
poetry add <package-name>
```

## Measure CPU usage

Make sure to start a poetry shell with by running `poetry shell`.
You can also run all commands outside a poetry shell, as long as you specify `poetry run` before the command.

I have implemented a simple wrapper script, the `cli.py`, 
as a helper to illustrate the use of the tools mentioned here,
and to allow for quick experimentation with different parameters. 
It is an easy way to get a quick overview about the code performance. 

Once you have a better clue and know you are heading to find
the sweet spot of **your hypothesis**, move on to use the 
profiling tools directly from the command line.

> NOTE: To run the examples, make sure the `PYTHONPATH` variable includes the `src` directory. Run `export PYTHONPATH=./src`.

### Overall cpu usage


Use the `cli.py` commands available to get the cpu usage report. 
The reports generated with `cProfile` are show graphically via `snakeviz`, 
directly in your default browser. Alternatively, when `--pretty` flag is given, 
`pyinstrument` is used to show an interactive condensed list,
which can also be exported to a `html` file.


Examples:

Measure using `cProfile`:

```bash
python cli.py cpu --save --vis
```

The `html` output files will be saved in the `.profile_data` directory. 
You can use any browser to render the results.


With the profile information saved to a file, you can trigger `snakeviz` at any time to render the contents again. 

```bash
snakeviz <full-path-to-output-file>
```

For easy access to the profiling output information, you can print them to the stdout at any time with `stats` option from the `cli` script. Examples:

```bash
python cli.py stats --file <full-path-to-output-file>
python cli.py stats --file <full-path-to-output-file> --sortby TIME --limit 5
python cli.py stats --file <full-path-to-output-file> --sortby TIME --limit 10 --callers
```


> The `cli stats` is a simple wrapper around the [`pstats`](https://docs.python.org/3/library/profile.html#module-pstats) python module.

To get a more condensed view on the result while using a statistical profiler, run the evaluation via `pyInstrument`:

```bash
cli.py cpu --save --vis --pretty
```



### Line-by-line cpu usage

`cProfile` only times explicit function calls, but not special methods called because of language syntax. So, many expensive computations are not considered by `cProfile` simply because there is not explicity function call.

`LineProfiler` can be given **specific functions** to evaluate, and it will time the execution of each individual line inside those functions.

To check the cpu usage in target functions, you must **decorate** the functions with the built-in `@profile` and run `kernprof` + `line_profiler`. For example:


Assuming `cli.py <task>` would trigger a time-consuming task in your application script:

```bash
python -m kernprof -l -v cli.py <task>
```

> NOTE: the `-l` option instruct `kernprof` to collect information on a line-by-line basis, as opposed to function level.
> `-v` toggles view mode to output information on the console.
> Without `-v`, you only get the result
> saved in a file, which you can inspect later using `line_profile` module.

By default `kernprof` saves the output in a file in the current directory and uses then input script name as `<script-to-profile>.py.lprof`. Use `-o` option to change the output file name.

How it works?

`kernprof` will create an instance of `LineProfiler` and insert it into the `__builtins__` namespace with the name **`profile`**. It has been written to be used as a **decorator**. So in your script, you have to decorate the functions you want to profile with `@profile`. Remember to **not import** anything for at this point. If you are editing the code in a rich IDE, it will probably complain about the unrecognized profile decorator, but you should not care about it for now.


Since the output is also exported to a file, you can than verify the results with:

```bash
python -m line_profiler cli.py.lprof
```

Alternatively, you can also use `pstats.Stats()` to read the results.

```bash
python -m pstats cli.py.lprof
```


## Measure Memory usage


The `mprof` utility samples the memory usage over time. Much like probabilistic cpu profilers it does not have a high impact on the code performance.
It can also plot a flame chart regarding the results.

### Overall time based memory usage

Overall time based memory usage can be checked via `cli.py`, via `mem` parameter and its options.
For example:

```bash
cli.py mem --exp <output-filename>.dat
```

This will generate a line chart for visualization and save the data into the desired file,
which can be later used to plot the chart again.

For more detailed information or more flexibility, use `mprof` directly on the command line.


### Function level memory usage

If you want to check **function level** memory usage, first decorate the target functions with `@profile` decorator. **Do not import** any decorator otherwise the code breaks.

Then run:

```bash
python -m mprof run -o <path-to-output-file>.prof cli.py <task>
```

This command runs the python script and record memory usage along time, saving the results in a file in the current directory

The result can be plotted for better visualization when you execute the following command:

```bash
python -m mprof plot -t "Recorded rule memory usage" <path-to-output-file>.prof --backend MacOSX
```

When generating the chart, the time boundaries for each decorated function will be shown in the chart as different colored square brackets.

> TIP: check out `--flame` option for `mprof plot`

It is also possible to use `mprof` to evaluate memory usage of multiprocessing code. Check out the documentation. Spoiler, you need to use `include_children` flag in either the `@profile` decorator or as a command line argument to `mprof`.

### Line level memory usage

In order to evaluate the memory usage at a line level, you need to use the `memory_profiler` utility. Also, it is necessary to decorate the target function the `@profile` **decorator provider by this package**.

Then you can load the `memory_profiler` `@profiler` decorator in 2 ways:

- either importing it from `memory_profiler` module in the python modules where the target functions are implemented; or
- passing `-m memory_profiler` to the python interpreter when running the target code.

> TIP: If the performance tests are executed from a `jupyter` notebook, you don't need to decorate the code.
> The `jupyter` extension simplifies the process allowing for the evaluation of a function without the decorator.

```bash
python -m memory_profiler <path-to-target-python-script>
```

The memory usage off all decorated functions will be reported in the stdout/console.


Using the labels really helps us to understand at a fine-grained level where memory is being consumed.

> TIP: check the magic command `%memit`

> TIP: `memory_profiler` supports the `--pdb-mmem=XXX` flag to debugging a large process. The pdb debugger will be activated after the process exceeds `XXX` MB.

## `cli.py` examples 

```python
python cli.py cpu --save
python cli.py stats --file .profile_data/1000-300-cprof.prof
python cli.py cpu --pretty
python cli.py mem
python cli.py mem --save
python -m mprof plot -t 'Your title' .profile_data/1000-300.dot
```