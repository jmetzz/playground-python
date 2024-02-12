from typing import List, Dict


def solve_dfs_recursive(target: int, arr: List[int]) -> bool:
    """Solution using brute force recursion

    O(n^m) time
    O(m) space
    """
    if target == 0:
        return True
    if target < 0:
        return False
    for _, value in enumerate(arr):
        solution = solve_dfs_recursive(target - value, arr)
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
