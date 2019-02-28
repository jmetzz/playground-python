def all_fib(n):
    cache = [0] * n
    for i in range(n):
        print(f"{i}: {fib(i, cache)}")


def fib(n, cache):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    elif cache[n] > 0:
        return cache[n]

    cache[n] = fib(n - 1, cache) + fib(n - 2, cache)
    return cache[n]


if __name__ == "__main__":
    all_fib(1000)
