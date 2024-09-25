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


def solve_tabulation(target: int, arr: list[int]) -> list[int] | None:
    pass


if __name__ == "__main__":
    print(solve_tabulation(7, [5, 3, 4, 7]))
    print(solve_tabulation(8, [2, 3, 5]))
    print(solve_tabulation(8, [1, 4, 5]))
    print(solve_tabulation(100, [1, 2, 5, 25]))
