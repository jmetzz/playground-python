"""Write a function that takes in a target_sum and an array of numbers as argument.

The function should return a boolean indication whether it is possible to generate
the target_sum using numbers from the array.

You may use an element of the array as many times as needed.

You may assume that all input numbers are non-negative.
"""

from typing import Dict, List


def solve_recursive(target: int, arr: List[int]) -> bool:
    """Solution using brute force recursion

    O(n^m) time
    O(m) space
    """
    if target == 0:
        return True
    if target < 0:
        return False
    for _, value in enumerate(arr):
        solution = solve_recursive(target - value, arr)
        if solution is True:
            return True
    return False


def solve_memoization(target: int, arr: List[int], cache: Dict[int, bool]) -> bool:
    """Solution using memoization

    O(m * n) time
    O(m) space
    """
    if target in cache:
        return cache[target]
    if target == 0:
        return True
    if target < 0:
        return False

    for _, value in enumerate(arr):
        new_target = target - value
        cache[new_target] = solve_memoization(new_target, arr, cache)
        # early stop if a solution is already found
        if cache[new_target]:
            return True
    return False
