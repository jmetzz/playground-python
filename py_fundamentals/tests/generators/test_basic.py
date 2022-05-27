"""
This module is not actually comprised of unit tests,
but rather a place to experiment with the code
"""

from generators.basic import fib_list


def test_fibonacci_list():
    """
    Examples to measure time and memory usage
    >>> %timeit test_fibonacci_list()
    332 ms ± 13.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

    >>> %memit test_fibonacci_list()
    peak memory: 492.82 MiB, increment: 441.75 MiB
    """
    for _ in fib_list(100_000):
        pass


def test_fib_list_div_by():
    len([n for n in fib_gen(100_000) if n % 3 == 0])


def test_fibonacci_gen():
    """
    Examples to measure time and memory usage
    >>> %timeit test_fibonacci_gen()
    126 ms ± 905 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

    >>> %memit test_fibonacci_gen()
    peak memory: 51.13 MiB, increment: 0.00 MiB
    """
    for _ in fib_gen(100_000):
        pass


def test_fib_gen_div_by():
    sum(n for n in fib_gen(100_000) if n % 3 == 0)
