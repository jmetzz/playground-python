"""https://leetcode.com/problems/spiral-matrix/description/
54. Spiral Matrix
Medium

Given an m x n matrix, return all elements of the matrix in spiral order.

Example 1:
Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [1,2,3,6,9,8,7,4,5]

Example 2:
Input: matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
Output: [1,2,3,4,8,12,11,10,9,5,6,7]


Constraints:

m == matrix.length
n == matrix[i].length
1 <= m, n <= 10
-100 <= matrix[i][j] <= 100
"""


def spiral_order(matrix: list[list[int]]) -> list[int]:
    """Iterate over the matrix in a spiral order: right, down, left, and up.
    Stop Condition:
        The iteration continues until the boundaries cross each other,
        i.e., top > bottom or left > right.
        This means that all elements have been visited in spiral order.
    """
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    answer = []
    while top <= bottom and left <= right:
        # Traverse from left to right
        for i in range(left, right + 1):
            answer.append(matrix[top][i])
        top += 1  # move down since top row has been completely traversed.

        # Traverse from top to bottom
        for i in range(top, bottom + 1):
            answer.append(matrix[i][right])
        right -= 1  # Right column fully traversed

        # Conditionally traverse from right to left
        if top <= bottom:
            for i in range(right, left - 1, -1):
                answer.append(matrix[bottom][i])
            bottom -= 1  # Bottom row fully traversed

        # Conditionally traverse from bottom to top
        if left <= right:
            for i in range(bottom, top - 1, -1):
                answer.append(matrix[i][left])
            left += 1  # Left column fully traversed

    return answer


print(spiral_order([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
