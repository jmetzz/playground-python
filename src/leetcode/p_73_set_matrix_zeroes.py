"""
https://leetcode.com/problems/set-matrix-zeroes/description

73. Set Matrix Zeroes
Medium

Given an m x n integer matrix matrix, if an element is 0,
set its entire row and column to 0's.

You must do it in place.



Example 1:
Input: matrix = [[1,1,1],[1,0,1],[1,1,1]]
Output: [[1,0,1],[0,0,0],[1,0,1]]

Example 2:
Input: matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
Output: [[0,0,0,0],[0,4,5,0],[0,3,1,0]]


Constraints:
m == matrix.length
n == matrix[0].length
1 <= m, n <= 200
-2^31 <= matrix[i][j] <= 2^31 - 1


Follow up:
A straightforward solution using O(mn) space is probably a bad idea.
A simple improvement uses O(m + n) space, but still not the best solution.
Could you devise a constant space solution?
"""

from data_structures.matrices import serialize


def set_zeroes_1(matrix: list[list[int]]) -> None:
    """
    Do not return anything, modify matrix in-place instead.
    """
    n, m = len(matrix), len(matrix[0])
    zeros = []
    for row in range(n):
        for col in range(m):
            if matrix[row][col] == 0:
                zeros.append((row, col))
    for row, col in zeros:
        for c in range(m):
            matrix[row][c] = 0
        for r in range(n):
            matrix[r][col] = 0


def set_zeroes_2(matrix: list[list[int]]) -> None:
    """
    Do not return anything, modify matrix in-place instead.
    """
    n, m = len(matrix), len(matrix[0])
    zero_rows = set()
    zero_cols = set()
    for row in range(n):
        for col in range(m):
            if matrix[row][col] == 0:
                zero_rows.add(row)
                zero_cols.add(col)
    for row in range(n):
        for col in range(m):
            if row in zero_rows or col in zero_cols:
                matrix[row][col] = 0


if __name__ == "__main__":
    matrix = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    set_zeroes_1(matrix)
    print(serialize(matrix))

    matrix = [[0, 1, 2, 0], [3, 4, 5, 2], [1, 3, 1, 5]]
    set_zeroes_2(matrix)
    print(serialize(matrix))
