"""This module is not actually comprised of unit tests,
but rather a place to experiment with the code
"""

from py_fundamentals.py_generators_example import fib_gen, fib_list


def test_fibonacci_list():
    for _ in fib_list(100_000):
        pass


def test_fib_list_div_by():
    len([n for n in fib_gen(100_000) if n % 3 == 0])


def test_fibonacci_gen():
    for _ in fib_gen(100_000):
        pass


def test_fib_gen_div_by():
    sum(n for n in fib_gen(100_000) if n % 3 == 0)
