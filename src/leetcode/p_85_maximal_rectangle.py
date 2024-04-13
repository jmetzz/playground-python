"""
https://leetcode.com/problems/maximal-rectangle/description

85. Maximal Rectangle
Hard

Given a rows x cols binary matrix filled with 0's and 1's, find the largest rectangle
containing only 1's and return its area.



Example 1:


Input: matrix = [
                ["1","0","1","0","0"],
                ["1","0","1","1","1"],
                ["1","1","1","1","1"],
                ["1","0","0","1","0"]]
Output: 6
Explanation: The maximal rectangle is shown in the above picture.

Example 2:
Input: matrix = [["0"]]
Output: 0

Example 3:
Input: matrix = [["1"]]
Output: 1


Constraints:

rows == matrix.length
cols == matrix[i].length
1 <= row, cols <= 200
matrix[i][j] is '0' or '1'.

"""

from typing import List


def maximal_rectangle(matrix: List[List[str]]) -> int:
    rows, cols = len(matrix), len(matrix[0])
    max_area = 0
    stack = []  # mono stack to keep (heigh of 1s, start idx), a series of heights in non-decreasing order
    heights = [0] * cols
    for row in range(rows):
        # column prefix sum of contiguous rectangle heights
        for col in range(cols):
            if matrix[row][col] == "0":
                heights[col] = 0
            else:
                heights[col] += 1

        # process the heights with a monotonic increasing stack.
        # the start variable keeps the column where the current
        # rectangle started (the index of the left most non zero element)
        stack.clear()
        for col in range(cols):
            start = col
            while stack and stack[-1][0] > heights[col]:
                # If the current height is less than the height at the top of the stack,
                # the stack is popped until a height that is less than or equal to
                # the current height is found. Each pop represents the end of a potential
                # maximal rectangle that extends from the index where the height was
                # first encountered (recorded as the second element in the tuple in the stack)
                # to the current index.
                prev_h, start = stack.pop()  # pop potential rectangle start idx
                candidate_area = prev_h * (col - start)
                if candidate_area > max_area:
                    max_area = candidate_area

            stack.append((heights[col], start))

        while stack:
            prev_h, start = stack.pop()  # pop potential rectangle start idx
            candidate_area = prev_h * (cols - start)
            if candidate_area > max_area:
                max_area = candidate_area
    return max_area


grid = [
    ["1", "0", "1", "0", "0"],
    ["1", "0", "1", "1", "1"],
    ["1", "1", "1", "1", "1"],
    ["1", "0", "0", "1", "0"],
]


maximal_rectangle(grid)
