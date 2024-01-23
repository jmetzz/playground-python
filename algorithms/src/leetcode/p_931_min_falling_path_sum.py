"""
https://leetcode.com/problems/minimum-falling-path-sum/

Given an n x n array of integers matrix, return the minimum sum of any falling path through matrix.

A falling path starts at any element in the first row and chooses the element in the next row
that is either directly below or diagonally left/right. Specifically, the next element
from position (row, col) will be (row + 1, col - 1), (row + 1, col), or (row + 1, col + 1).

Input: matrix = [[2,1,3],[6,5,4],[7,8,9]]
Output: 13
Explanation: There are two falling paths with a minimum sum as shown.

Input: matrix = [[-19,57],[-40,-5]]
Output: -59
Explanation: The falling path with a minimum sum is shown.


Constraints:

n == matrix.length == matrix[i].length
1 <= n <= 100
-100 <= matrix[i][j] <= 100

"""
from typing import List


class MinFallingPathSumViaRecursion:
    def solve(self, matrix: List[List[int]]) -> int:
        n = len(matrix)
        costs = [0] * n
        for col in range(n):
            costs[col] = matrix[n - 1][col] + min(
                self._solve(matrix, n, n - 2, col - 1),
                self._solve(matrix, n, n - 2, col),
                self._solve(matrix, n, n - 2, col + 1)
            )
        return min(costs)

    def _solve(self, matrix, n, row, col) -> int:
        if col < 0 or col >= n:
            return 999
        if row < 0 or row >= n:
            return 0
        if row == 0:
            return matrix[row][col]
        return matrix[row][col] + min(
            self._solve(matrix, n, row - 1, col - 1),
            self._solve(matrix, n, row - 1, col),
            self._solve(matrix, n, row - 1, col + 1)
        )


class MinFallingPathSumViaDP:
    def solve(self, matrix: List[List[int]]) -> int:
        n = len(matrix)
        # one line only, return the smaller element
        if n == 1:
            return min(matrix[0])
        sentinel = float('inf')
        # two extra columns to accommodate the boundary cases
        # both to the left and to the right
        dp_table_line = [sentinel] * (n + 2)

        for row in range(1, n, 1):
            dp_table_line[1:n + 1] = matrix[row - 1]
            for col in range(n):
                # target_costs[col] = matrix[row][col] + min(
                matrix[row][col] = matrix[row][col] + min(
                    dp_table_line[col],  # left
                    dp_table_line[col + 1],  # center
                    dp_table_line[col + 2]  # right
                )
        # the smaller element in the last line has the min cost
        return min(matrix[n - 1])

