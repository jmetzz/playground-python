"""
https://leetcode.com/problems/count-submatrices-with-all-ones/description/

1504. Count Submatrices With All Ones
Medium
Explanation: https://youtu.be/cxzoOv3iwM8?si=IZHZJeftJs1RrHWm

Given an m x n binary matrix mat, return the number of submatrices that have all ones.

Input: mat = [[1,0,1],[1,1,0],[1,1,0]]
Output: 13
Explanation:
There are 6 rectangles of side 1x1.
There are 2 rectangles of side 1x2.
There are 3 rectangles of side 2x1.
There is 1 rectangle of side 2x2.
There is 1 rectangle of side 3x1.
Total number of rectangles = 6 + 2 + 3 + 1 + 1 = 13.
Example 2:


Input: mat = [[0,1,1,0],[0,1,1,1],[1,1,1,0]]
Output: 24
Explanation:
There are 8 rectangles of side 1x1.
There are 5 rectangles of side 1x2.
There are 2 rectangles of side 1x3.
There are 4 rectangles of side 2x1.
There are 2 rectangles of side 2x2.
There are 2 rectangles of side 3x1.
There is 1 rectangle of side 3x2.
Total number of rectangles = 8 + 5 + 2 + 4 + 2 + 2 + 1 = 24.


Constraints:

1 <= m, n <= 150
mat[i][j] is either 0 or 1.

Hint 1
For each row i, create an array nums where: if mat[i][j] == 0 then nums[j] = 0
else nums[j] = nums[j-1] +1.

Hint 2
In the row i, number of rectangles between column j and k(inclusive) and ends in row i,
is equal to SUM(min(nums[j, .. idx])) where idx go from j to k. Expected solution is O(n^3).

"""

from typing import List


def num_submatrices_of_ones__brute_force(matrix: List[List[int]]) -> int:
    """
    Time: O(n^2 * m^2)
    Space: O(m * n)

    The idea:
    if not matrix:
        return 0

    # first step: precompute the 2D prefix sums

    # second step:
    rows, cols  = len(matrix),len(matrix[0])
    counter = 0
    for upper_row in range(rows):
        for left_col in range(cols):
            for lower_row in range(upper_row, rows):
                for right_col in range(left_col, cols):
                    # check if all elements in the submatrix are 1s:
                    # In order to calculate this in O(1), we can use the precomputed 2D prefix sum
                    # if sum(upper_row, left_col, lower_row, right_col) ==
                    # (right_col - left_col + 1) * (upper_row - lower_row + 1):
                    #     counter += 1

    return counter
    """
    raise NotImplementedError()


def num_submatrices_of_ones__dp(matrix: List[List[int]]) -> int:
    counter = 0
    num_rows, num_cols = len(matrix), len(matrix[0])
    # Time complexity:
    # O(n) * O(m + n * m)
    # O(n^2 * m)
    # Space complexity:
    # O(m)
    full_col = [1] * num_cols

    for upper_row in range(num_rows):  # O(n)
        for c in range(num_cols):  # O(m)
            full_col[c] = 1

        for lower_row in range(upper_row, num_rows):  # O(n)
            # how many submatrices (upper_row, ?, lower_row, ?) are full of 1s?
            for col in range(num_cols):  # O(m) Time step
                if matrix[lower_row][col] == 0:
                    full_col[col] = 0
            curr_len = 0
            for c in full_col:  # O(m)
                if c == 1:
                    curr_len += 1
                else:
                    curr_len = 0
                counter += curr_len

    return counter


def num_submatrices_of_ones__mono_stack(matrix: List[List[int]]) -> int:
    """
    Intuition:
    stack the matrix row by row to build a "histogram model", where each
    element in column i, represents the sum of consecutive 1s in that column.
    For example:
        mat = [[1,0,1],
               [1,1,0],
               [1,1,0]]

        becomes

        histogram = [[1,0,1],
                     [2,1,0],
                     [3,2,0]]

    The second phase is to compute the counter, which is done by the helper function.
    Traverse the stacked histogram by row and compute the number of all-1 submatrices
    at each position [i, j].

    """

    def helper(hist: List[int]) -> int:
        sums = [0] * len(hist)
        stack = []  # mono-stack of indices of non-decreasing height

        for i in range(len(hist)):
            while stack and hist[stack[-1]] >= hist[i]:
                stack.pop()

            if stack:
                pre_index = stack[-1]
                sums[i] = sums[pre_index]
                sums[i] += hist[i] * (i - pre_index)
            else:
                sums[i] = hist[i] * (i + 1)
            stack.append(i)

        return sum(sums)

    num_rows, num_cols = len(matrix), len(matrix[0])
    counter = 0

    histogram = [0] * num_cols
    for row in range(num_rows):
        for col in range(num_cols):
            if matrix[row][col] == 0:
                histogram[col] = 0
            else:
                histogram[col] + 1
        counter += helper(histogram)

    return counter
