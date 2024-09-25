"""https://leetcode.com/problems/diagonal-traverse/description/

498. Diagonal Traverse
Medium

Given an m x n matrix mat, return an array of all the elements of the array
in a diagonal order.


Example 1:
Input: mat = [[1,2,3],[4,5,6],[7,8,9]]
Output: [1,2,4,7,5,3,6,8,9]

Example 2:
Input: mat = [[1,2],[3,4]]
Output: [1,2,3,4]


Constraints:
m == mat.length
n == mat[i].length
1 <= m, n <= 104
1 <= m * n <= 104
-105 <= mat[i][j] <= 105
"""

from collections import defaultdict


def find_diagonal_natural_order(grid: list[list[int]]) -> list[int]:
    if not grid or not grid[0]:
        return []
    n, m = len(grid), len(grid[0])

    answer = []
    for diagonal in range(m + n - 1):
        # identify the head of the diagonal
        if diagonal < m:
            row = 0
            col = diagonal
        else:
            row = diagonal - m + 1
            col = m - 1

        # Iterate until one of the indices goes out of bounds
        # Take note of the index math to go down the diagonal
        while row < n and col > -1:
            answer.append(grid[row][col])
            row += 1
            col -= 1
    return answer


def find_diagonal_order_zigzag(grid: list[list[int]]) -> list[int]:
    if not grid or not grid[0]:
        return []
    n, m = len(grid), len(grid[0])

    answer = []
    for diagonal in range(m + n - 1):
        # iterate over all the elements in the first row and
        # the last column to cover all possible diagonal heads
        intermediate = []

        # identify the head of the diagonal
        if diagonal < m:
            row = 0
            col = diagonal
        else:
            row = diagonal - m + 1
            col = m - 1

        # Iterate until one of the indices goes out of bounds
        # Take note of the index math to go down the diagonal
        while row < n and col > -1:
            intermediate.append(grid[row][col])
            row += 1
            col -= 1

        # Reverse even numbered diagonals
        if diagonal % 2 == 0:
            answer.extend(intermediate[::-1])
        else:
            answer.extend(intermediate)
    return answer


def find_diagonal_order_simple(grid: list[list[int]]) -> list[int]:
    """Sum of diagonal indices determines the diagonal

    The key here is to realize that the sum of indices of elements
    is always are equal for all elements the same diagonal.

    Consider this grid:
        [[1, 2, 3]
         [4, 5, 6]
         [7, 8, 9]]

    The diagonal 0: [0, 0]
    The diagonal 1: [1, 0] and [0, 1]
    The diagonal 2: [2, 0] and [1, 1] and [0, 2]
    ...

    With the diagonals figured out, one need to model the 'zig zag'
    of the diagonar for this problem. This is simple and can be
    achieve in several ways.
    I use the % opereration to check if the diagonal is even,
    and if yes, invert the selection of elements before adding
    to the final answer.
    """
    if not grid or not grid[0]:
        return []
    diagonal_map = defaultdict(list)
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            d = row + col
            diagonal_map[d].append(grid[row][col])
    answer = []
    for d in diagonal_map:
        if d % 2 == 0:
            answer.extend(diagonal_map[d][::-1])
        else:
            answer.extend(diagonal_map[d])
    return answer


if __name__ == "__main__":
    grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    print(f"Diagonal natural order: {find_diagonal_natural_order(grid)}")
    print(f"Diagonal zig-zag order: {find_diagonal_order_zigzag(grid)}")
    print(f"Diagonal zig-zag order: {find_diagonal_order_simple(grid)}")
