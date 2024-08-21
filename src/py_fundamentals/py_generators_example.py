from typing import List


def fib_list(num_items: List[int]) -> List[int]:
    """Examples to measure time and memory usage
    >>> %timeit fib_list()
    332 ms ± 13.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

    >>> %memit fib_list()
    peak memory: 492.82 MiB, increment: 441.75 MiB

    """
    numbers = []
    a, b = 0, 1
    while len(numbers) < num_items:
        numbers.append(a)
        a, b = b, a + b
    return numbers


def fib_gen(num_items):
    a, b = 0, 1
    while num_items:
        yield a
        a, b = b, a + b
        num_items -= 1


def all_fib_gen():
    i, j = 0, 1
    while True:
        yield j
        i, j = j, i + j
