"""
https://leetcode.com/problems/cherry-pickup-ii/description/

1463. Cherry Pickup II
Hard

You are given a rows x cols matrix grid representing a field of cherries where grid[i][j] represents the number of cherries that you can collect from the (i, j) cell.
You have two robots that can collect cherries for you:

Robot #1 is located at the top-left corner (0, 0), and
Robot #2 is located at the top-right corner (0, cols - 1).
Return the maximum number of cherries collection using both robots by following the rules below:

From a cell (i, j), robots can move to cell (i + 1, j - 1), (i + 1, j), or (i + 1, j + 1).
When any robot passes through a cell, It picks up all cherries, and the cell becomes an empty cell.
When both robots stay in the same cell, only one takes the cherries.
Both robots cannot move outside of the grid at any moment.
Both robots should reach the bottom row in grid.


Example 1:
Input: grid = [[3,1,1],[2,5,1],[1,5,5],[2,1,1]]
Output: 24
Explanation: Path of robot #1 and #2 are described in color green and blue respectively.
Cherries taken by Robot #1, (3 + 2 + 5 + 2) = 12.
Cherries taken by Robot #2, (1 + 5 + 5 + 1) = 12.
Total of cherries: 12 + 12 = 24.

Example 2:
Input: grid = [[1,0,0,0,0,0,1],[2,0,0,0,0,3,0],[2,0,9,0,0,0,0],[0,3,0,5,4,0,0],[1,0,2,3,0,0,6]]
Output: 28
Explanation: Path of robot #1 and #2 are described in color green and blue respectively.
Cherries taken by Robot #1, (1 + 9 + 5 + 2) = 17.
Cherries taken by Robot #2, (1 + 3 + 4 + 3) = 11.
Total of cherries: 17 + 11 = 28.


Constraints:
rows == grid.length
cols == grid[i].length
2 <= rows, cols <= 70
0 <= grid[i][j] <= 100
"""

from functools import cache
from typing import List


def cherryPickup_two_robots_dfs(grid: List[List[int]]) -> int:
    rows = len(grid)
    if rows == 0:
        return 0
    cols = len(grid[0])

    # tree actions possible: down, left-down, right-down
    # so we offset the column index by these delta values
    # this is used to try improving readability, as opposed to
    # using 'for k in range(max(0, col - 1), min(col + 2, cols))'
    delta = [0, -1, 1]

    def dfs(row: int, robot_1: int, robot_2: int) -> int:
        """
        the row is constant in each call, and the robot_* here represent
        the current column the respective robot is on
        """
        if row == rows:
            # got to the base of the matrix
            return 0
        if robot_1 < 0 or robot_2 < 0 or robot_1 >= cols or robot_2 >= cols:
            # out of bounds base case
            return float('-inf')

        answer = 0
        for k in range(3):
            for r in range(3):
                answer = max(answer, dfs(row + 1, robot_1 + delta[k], robot_2 + delta[r]))

        if robot_1 == robot_2:
            # collision case
            # arbitrarily take robot_1, and robot_2 takes 0
            answer += grid[row][robot_1]
        else:
            answer += grid[row][robot_1] + grid[row][robot_2]
        return answer

    return dfs(row=0, robot_1=0, robot_2=cols - 1)


def cherryPickup_two_robots_dfs_memoization(grid: List[List[int]]) -> int:
    rows = len(grid)
    if rows == 0:
        return 0
    cols = len(grid[0])

    # tree actions possible: down, left-down, right-down
    # so we offset the column index by these delta values
    # this is used to try improving readability, as opposed to
    # using 'for k in range(max(0, col - 1), min(col + 2, cols))'
    delta = (0, -1, 1)

    @cache
    def dfs(row: int, robot_1: int, robot_2: int) -> int:
        """
        the row is constant in each call, and the robot_* here represent
        the current column the respective robot is on
        """
        if row == rows:
            # got to the base of the matrix
            return 0
        if robot_1 < 0 or robot_2 < 0 or robot_1 >= cols or robot_2 >= cols:
            # out of bounds base case
            return float('-inf')

        answer = 0
        for k in range(3):
            for r in range(3):
                answer = max(answer, dfs(row + 1, robot_1 + delta[k], robot_2 + delta[r]))

        if robot_1 == robot_2:
            # collision case
            # arbitrarily take robot_1, and robot_2 takes 0
            answer += grid[row][robot_1]
        else:
            answer += grid[row][robot_1] + grid[row][robot_2]

        return answer

    return dfs(row=0, robot_1=0, robot_2=cols - 1)


if __name__ == '__main__':
    grid_1 = [[3, 1, 1], [2, 5, 1], [1, 5, 5], [2, 1, 1]]
    grid_2 = [[1, 0, 0, 0, 0, 0, 1],
              [2, 0, 0, 0, 0, 3, 0],
              [2, 0, 9, 0, 0, 0, 0],
              [0, 3, 0, 5, 4, 0, 0],
              [1, 0, 2, 3, 0, 0, 6]]

    print(cherryPickup_two_robots_dfs_memoization(grid_2))
    grid_3 = [[8, 8, 10, 9, 1, 7], [8, 8, 1, 8, 4, 7], [8, 6, 10, 3, 7, 7], [3, 0, 9, 3, 2, 7], [6, 8, 9, 4, 2, 5],
              [1, 1, 5, 8, 8, 1], [5, 6, 5, 2, 9, 9], [4, 4, 6, 2, 5, 4], [4, 4, 5, 3, 1, 6], [9, 2, 2, 1, 9, 3]]

    print(cherryPickup_two_robots_dfs(grid_3))  # 146 -- slow
    print(cherryPickup_two_robots_dfs_memoization(grid_3))  # 146 -- fast
