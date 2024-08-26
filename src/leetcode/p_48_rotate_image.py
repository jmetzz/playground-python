"""
48. Rotate Image
Medium

You are given an n x n 2D matrix representing an image,
rotate the image by 90 degrees (clockwise).

You have to rotate the image in-place, which means you have to
modify the input 2D matrix directly.

DO NOT allocate another 2D matrix and do the rotation.



Example 1:
Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [[7,4,1],[8,5,2],[9,6,3]]

Example 2:
Input: matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
Output: [[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]


Constraints:
n == matrix.length == matrix[i].length
1 <= n <= 20
-1000 <= matrix[i][j] <= 1000
"""


def rotate_image(matrix: list[list[int]]) -> None:
    """
    Do not return anything, modify matrix in-place instead.

    layered based approach
    """
    left, right = 0, len(matrix) - 1
    while left < right:
        for offset in range(right - left):
            top, bottom = left, right  # for readability

            topleft = matrix[top][left + offset]  # save top left element

            # move the bottom left to the top left place
            matrix[top][left + offset] = matrix[bottom - offset][left]

            # move the bottom right into the bottom left position
            matrix[bottom - offset][left] = matrix[bottom][right - offset]

            # move the top right into the bottom right
            matrix[bottom][right - offset] = matrix[top + offset][right]

            # restore temp element at top right
            matrix[top + offset][right] = topleft
        left += 1
        right -= 1


if __name__ == "__main__":
    matrix = [[5, 1, 9, 11], [2, 4, 8, 10], [13, 3, 6, 7], [15, 14, 12, 16]]
    rotate_image(matrix)
    print(matrix)  # Output: [[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]
