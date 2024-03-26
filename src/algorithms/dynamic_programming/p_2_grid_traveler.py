"""
Considering a 2d grid of mxn, how many ways can you travel from the top left corner for the bottom right corner?
The only actions allowed are:
1. move down
2. move right

"""

from typing import Dict


def solve_recursive(m: int, n: int) -> int:
    # base
    if m == 0 or n == 0:
        # invalid grid
        return 0
    if m == 1 and n == 1:
        # arrived
        return 1
    return solve_recursive(m - 1, n) + solve_recursive(m, n - 1)


def solve_memoization(m: int, n: int, cache: Dict) -> int:
    # cache check
    key = (m, n)
    if key in cache:
        return cache[key]

    # base # invalid grid
    if m == 0 or n == 0:
        return 0

    # arrived
    if m == 1 and n == 1:
        return 1

    cache[key] = solve_memoization(m - 1, n, cache) + solve_memoization(m, n - 1, cache)
    return cache[key]


if __name__ == "__main__":
    memo = dict()
    value = solve_memoization(2, 3, memo)
    print(value)
    print(memo)

    value = solve_memoization(18, 18, memo)  # 2333606220
    print(value)
