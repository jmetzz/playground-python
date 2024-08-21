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

from typing import List, Optional


def solve_tabulation(target: int, arr: List[int]) -> Optional[List[int]]:
    pass


if __name__ == "__main__":
    print(solve_tabulation(7, [2, 3]))
    print(solve_tabulation(7, [5, 3, 4, 7]))
    print(solve_tabulation(7, [2, 4]))
    print(solve_tabulation(8, [2, 3, 5]))
