"""
https://leetcode.com/problems/cherry-pickup/description/

741. Cherry Pickup
Hard

You are given an n x n grid representing a field of cherries, each cell is one of three possible integers.

0 means the cell is empty, so you can pass through,
1 means the cell contains a cherry that you can pick up and pass through, or
-1 means the cell contains a thorn that blocks your way.
Return the maximum number of cherries you can collect by following the rules below:

Starting at the position (0, 0) and reaching (n - 1, n - 1) by moving right or down through valid path cells (cells with value 0 or 1).
After reaching (n - 1, n - 1), returning to (0, 0) by moving left or up through valid path cells.
When passing through a path cell containing a cherry, you pick it up, and the cell becomes an empty cell 0.
If there is no valid path between (0, 0) and (n - 1, n - 1), then no cherries can be collected.

Example 1:
Input: grid = [[0,1,-1],[1,0,-1],[1,1,1]]
Output: 5
Explanation: The player started at (0, 0) and went down, down, right right to reach (2, 2).
4 cherries were picked up during this single trip, and the matrix becomes [[0,1,-1],[0,0,-1],[0,0,0]].
Then, the player went left, up, up, left to return home, picking up one more cherry.
The total number of cherries picked up is 5, and this is the maximum possible.

Example 2:
Input: grid = [[1,1,-1],[1,-1,1],[-1,1,1]]
Output: 0


Constraints:
n == grid.length
n == grid[i].length
1 <= n <= 50
grid[i][j] is -1, 0, or 1.
grid[0][0] != -1
grid[n - 1][n - 1] != -1
"""
from typing import List


def cherryPickup_one_robot_dp_full_table(grid: List[List[int]]) -> int:
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


if __name__ == '__main__':
    grid_1 = [[3, 1, 1], [2, 5, 1], [1, 5, 5], [2, 1, 1]]
    grid_2 = [[1, 0, 0, 0, 0, 0, 1],
              [2, 0, 0, 0, 0, 3, 0],
              [2, 0, 9, 0, 0, 0, 0],
              [0, 3, 0, 5, 4, 0, 0],
              [1, 0, 2, 3, 0, 0, 6]]
    grid_3 = [[0, 1, -1], [1, 0, -1], [1, 1, 1]]
    grid_4 = [[1, 1, -1], [1, -1, 1], [-1, 1, 1]]
    print(cherryPickup_one_robot_dp_full_table(grid_1))
    print(cherryPickup_one_robot_dp_full_table(grid_2))
    print(cherryPickup_one_robot_dp_full_table(grid_3))  # 5
    print(cherryPickup_one_robot_dp_full_table(grid_4))  # 0
