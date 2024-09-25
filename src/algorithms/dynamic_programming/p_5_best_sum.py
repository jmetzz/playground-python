"""Implement a function that takes in a target_sum and an
array of numbers (v_i >= 0) as arguments.

The function should return an array containing the
shortest combination of elements that add up
to exactly the target_sum. If there is any tie for
the shortest combination, return any of the shortest.

If there is no combination that adds up the target-sum,
then return None.

Examples:
--------
    best_sum(7, [5, 3, 4, 7]) -> [7]
    best_sum(7, [2, 4]) -> None
    best_sum(8, [2, 3, 5]) -> [3, 5]
    best_sum(8, [1, 4, 5]) -> [4, 4]
    best_sum(100, [1, 3, 5, 25]) -> [25, 25, 25, 25]
    best_sum(0, [2, 3, 5]) -> []

"""


def solve_recursive(target: int, arr: list[int]) -> list[int] | None:
    """Solution using brute force recursion
    Exhaustive search exploring the full recursion tree

    O() time
    O() space
    """
    if target == 0:
        return []
    if target < 0:
        return None

    curr_best = None
    for value in arr:
        candidate = solve_recursive(target - value, arr)
        if candidate is None:
            continue
        if curr_best is None or len(candidate) + 1 < len(curr_best):
            curr_best = [value] + candidate
    return curr_best


def solve_memoization(target: int, arr: list[int], cache) -> list[int] | None:
    """Solution using brute force recursion
    Exhaustive search exploring the full recursion tree

    O(m^2 * n) time
    O(m^2) space
    """
    if target in cache:
        return cache[target]

    if target == 0:
        return []

    if target < 0:
        return None

    cur_best = None
    for value in arr:
        solution = solve_memoization(target - value, arr, cache)
        if solution is None:
            continue
        if cur_best is None or len(solution) < len(cur_best):
            cur_best = [value] + solution

    cache[target] = cur_best
    return cache[target]


if __name__ == "__main__":
    print(solve_recursive(7, [5, 3, 4, 7]))
    print(solve_recursive(8, [2, 3, 5]))
    print(solve_recursive(8, [1, 4, 5]))
    print(solve_recursive(70, [1, 2, 5, 25]))
    print("---")

    print(solve_memoization(7, [5, 3, 4, 7], {}))
    print(solve_memoization(8, [2, 3, 5], {}))
    print(solve_memoization(8, [1, 4, 5], {}))
    print(solve_memoization(100, [1, 2, 5, 25], {}))
