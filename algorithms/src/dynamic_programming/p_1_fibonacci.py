"""
The Fibonacci sequence is defined by the recurrence relation F(n) = F(n-1) + F(n-2),
with initial conditions F(0) = 0 and F(1) = 1.
"""
from typing import Dict, List


def fib_recursive(n: int) -> int:
    """Classic recursive implementation of fibonacci sequence.

    Time and space complexity are O(2^n).
    """
    if n == 0:
        return 0
    if n == 1:
        return 1
    return fib_recursive(n - 1) + fib_recursive(n - 2)


def fib_memoization(n: int, cache: Dict[int, int]) -> int:
    """Classic recursive implementation of fibonacci sequence.

    Time and space complexity are O(2^n).
    """
    if n in cache:
        return cache[n]
    if n == 0 or n == 1:
        cache[n] = n
        return n
    solution = fib_memoization(n - 1, cache) + fib_memoization(n - 2, cache)
    cache[n] = solution
    return solution


def fib_sequence(n: int) -> List[int]:
    """Generate the entire fibonacci sequence from 1 to n."""
    memo = {0: 0}
    _ = fib_memoization(n, memo)

    return sorted([value for _, value in memo.items()])


def fib_sequence_via_generator(num_items):
    a, b = 0, 1
    while num_items + 1:
        yield a
        a, b = b, a + b
        num_items -= 1


if __name__ == "__main__":
    # for i in range(8):
    #     print(fib_with_memoization(i))
    # print(fib_with_memoization(50))
    #
    print(fib_sequence(7))
    #
    # for i in range(8):
    #     print(fib_recursive(i))
    # print("---")
    #
    # for v in fib_sequence_via_generator(7):
    #     print(v)

    # print(fib_with_memoization(2, {}))
