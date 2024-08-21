"""A simplified version of the problem "1463. Cherry Pickup II"

You are given a rows x cols matrix grid representing a field of cherries
where grid[i][j] represents the number of cherries that you can collect from the (i, j) cell.

You have one robot that can collect cherries for you and it is located at the top-left corner (0, 0).

Return the maximum number of cherries collection using both robots by following the rules below:

From a cell (i, j), the robot can move to cell (i + 1, j - 1), (i + 1, j), or (i + 1, j + 1).
When the robot passes through a cell, it picks up all cherries, and the cell becomes an empty cell.
The robot cannot move outside of the grid at any moment.
The robot should reach the bottom row in grid.


Constraints:
rows == grid.length
cols == grid[i].length
2 <= rows, cols <= 70
0 <= grid[i][j] <= 100
"""

from typing import List


def cherryPickup_one_robot(grid: List[List[int]]) -> int:
    rows, cols = len(grid), len(grid[0])
    dp = [[0] * cols for i in range(rows)]
    dp[0] = grid[0]
    r, c = 0, 0  # the answer indices

    for i in range(1, rows):
        for j in range(cols):
            parent_value = dp[i - 1][j]
            for k in range(j - 1, j + 2):  # remember range second argument is not inclusive
                if k < 0 or k >= cols:
                    continue  # skip invalid positions
                dp[i][k] = max(dp[i][k], parent_value + grid[i][k])
                if dp[i][k] > dp[r][c]:
                    r, c = i, k
    return dp[r][c]


if __name__ == "__main__":
    grid_1 = [[3, 1, 1], [2, 5, 1], [1, 5, 5], [2, 1, 1]]
    grid_2 = [
        [1, 0, 0, 0, 0, 0, 1],
        [2, 0, 0, 0, 0, 3, 0],
        [2, 0, 9, 0, 0, 0, 0],
        [0, 3, 0, 5, 4, 0, 0],
        [1, 0, 2, 3, 0, 0, 6],
    ]

    grid_3 = [
        [8, 8, 10, 9, 1, 7],
        [8, 8, 1, 8, 4, 7],
        [8, 6, 10, 3, 7, 7],
        [3, 0, 9, 3, 2, 7],
        [6, 8, 9, 4, 2, 5],
        [1, 1, 5, 8, 8, 1],
        [5, 6, 5, 2, 9, 9],
        [4, 4, 6, 2, 5, 4],
        [4, 4, 5, 3, 1, 6],
        [9, 2, 2, 1, 9, 3],
    ]
    print(cherryPickup_one_robot(grid_1))
    print(cherryPickup_one_robot(grid_2))
    print(cherryPickup_one_robot(grid_3))
