"""Take a function that takes in a target_sum and an array of numbers (v_i >= 0) as arguments.

The function should return an array containing any combination of elements that add up
to exactly the target_sum. If there is no combination that adds up the target-sum,
then return None.

If there are multiple combinations possible, you may return a single on.

Example:
-------
    how_sum(7, [5, 3, 4, 7]) -> [7] or [3, 4]
    how_sum(7, [2, 4]) -> None
    how_sum(8, [2, 3, 5]) -> [3, 5] or [2, 2, 2, 2] or [3, 3, 2] ...
    how_sum(0, [2, 3, 5]) -> []

"""

from typing import Dict, List, Optional


def solve_recursive(target: int, arr: List[int]) -> Optional[List[int]]:
    """Solution using brute force recursion

    O(n^m * m) time
    O(m) space
    """
    if target == 0:
        return []
    if target < 0:
        return None

    for _, value in enumerate(arr):
        solution = solve_recursive(target - value, arr)
        if solution is not None:
            return [value] + solution
    return None


def solve_memoization(target: int, arr: List[int], cache: Dict[int, List[int]]) -> Optional[List[int]]:
    """Solution using memoization

    O(n * m^2) time
    O(m^2) space
    """
    if target in cache:
        return cache[target]
    if target == 0:
        return []
    if target < 0:
        return None

    for _, value in enumerate(arr):
        new_target = target - value
        cache[new_target] = solve_memoization(new_target, arr, cache)
        # early stop if a solution is already found
        if cache[new_target] is not None:
            return [value] + cache[new_target]
    return None


if __name__ == "__main__":
    print(solve_recursive(7, [2, 3]))
    print(solve_recursive(7, [5, 3, 4, 7]))
    print(solve_recursive(7, [2, 4]))
    print(solve_recursive(8, [2, 3, 5]))
    print("---")
    print(solve_memoization(7, [2, 3], {}))
    print(solve_memoization(7, [5, 3, 4, 7], {}))
    print(solve_memoization(7, [2, 4], {}))
    print(solve_memoization(8, [2, 3, 5], {}))
