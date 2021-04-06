def fib_list(num_items):
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


def test_fibonacci_list():
    """
    >>> %timeit test_fibonacci_list()
    332 ms ± 13.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

    >>> %memit test_fibonacci_list()
    peak memory: 492.82 MiB, increment: 441.75 MiB
    """
    for i in fib_list(100_000):
        pass


def test_fib_list_div_by():
    divisible_by_three = len([n for n in fib_gen(100_000) if n % 3 == 0])
    return divisible_by_three


def test_fib_gen_div_by():
    divisible_by_three = sum(n for n in fib_gen(100_000) if n % 3 == 0)
    return divisible_by_three


def test_fibonacci_gen():
    """
    >>> %timeit test_fibonacci_gen()
    126 ms ± 905 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

    >>> %memit test_fibonacci_gen()
    peak memory: 51.13 MiB, increment: 0.00 MiB
    """
    for i in fib_gen(100_000):
        pass
