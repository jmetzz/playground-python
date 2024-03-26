"""

https://leetcode.com/problems/number-of-submatrices-that-sum-to-target/description

1074. Number of Submatrices That Sum to Target
#Hard

Explanation: https://youtu.be/43DRBP2DUHg?si=uJ_BZOW3XALArhO_

Given a matrix and a target, return the number of non-empty submatrices that sum to target.

A submatrix x1, y1, x2, y2 is the set of all cells matrix[x][y] with x1 <= x <= x2 and y1 <= y <= y2.

Two submatrices (x1, y1, x2, y2) and (x1', y1', x2', y2') are different if they have some coordinate that is different:
for example, if x1 != x1'.

Example 1:
Input: matrix = [[0,1,0],[1,1,1],[0,1,0]], target = 0
Output: 4
Explanation: The four 1x1 submatrices that only contain 0.


Example 2:
Input: matrix = [[1,-1],[-1,1]], target = 0
Output: 5
Explanation: The two 1x2 submatrices, plus the two 2x1 submatrices, plus the 2x2 submatrix.

Example 3:
Input: matrix = [[904]], target = 0
Output: 0

Constraints:

1 <= matrix.length <= 100
1 <= matrix[0].length <= 100
-1000 <= matrix[i] <= 1000
-10^8 <= target <= 10^8
"""

from collections import defaultdict
from typing import List


class NumSubmatricesSumTarget:
    def solve_2d_grid(self, matrix: List[List[int]], target: int) -> int:
        """This is not really trivial,
        but is the slow solution.
        """
        num_rows, num_cols = len(matrix), len(matrix[0])
        sub_sum = [[0] * num_cols for _ in range(num_rows)]

        # compute the 2d grid prefix sub-matrices sum
        for row in range(num_rows):
            for col in range(num_cols):
                top = sub_sum[row - 1][col] if row > 0 else 0
                left = sub_sum[row][col - 1] if col > 0 else 0
                top_left = sub_sum[row - 1][col - 1] if min(row, col) > 0 else 0
                sub_sum[row][col] = matrix[row][col] + top + left - top_left

        # check if the sub-matrices sum add to the target value
        result = 0
        for r1 in range(num_rows):
            for r2 in range(r1, num_rows):
                for c1 in range(num_cols):
                    for c2 in range(c1, num_cols):
                        top = sub_sum[r1 - 1][c2] if r1 > 0 else 0
                        left = sub_sum[r2][c1 - 1] if c1 > 0 else 0
                        top_left = sub_sum[r1 - 1][c1 - 1] if min(r1, c1) > 0 else 0
                        cur_sum = sub_sum[r2][c2] - top - left + top_left
                        if cur_sum == target:
                            result += 1
        return result

    def solve_hashmap(self, matrix: List[List[int]], target: int) -> int:
        """
        Time complexity: O(m*n^2)
        Space complexity: O(n)
        """
        num_rows, num_cols = len(matrix), len(matrix[0])
        sub_sum = [[0] * num_cols for _ in range(num_rows)]

        # compute the 2d grid prefix sub-matrices sum
        for row in range(num_rows):
            for col in range(num_cols):
                top = sub_sum[row - 1][col] if row > 0 else 0
                left = sub_sum[row][col - 1] if col > 0 else 0
                top_left = sub_sum[row - 1][col - 1] if min(row, col) > 0 else 0
                sub_sum[row][col] = matrix[row][col] + top + left - top_left

        # check if the sub-matrices sum add to the target value
        result = 0
        for r1 in range(num_rows):
            for r2 in range(r1, num_rows):
                # Create a map object 'prefix_sum -> count' of sub-matrices that sum_up to target ending at colum "key"
                count = defaultdict(int)
                count[0] = 1  # base
                for c in range(num_cols):
                    # Check if the current sum minus the target value exists in the HashMap.
                    # If it does, increment the count by the frequency of that sum.
                    # Also, keep in mind cur_sum ending at r2 and column c,
                    # then adjust the value by subtracting  sub_sum[r1 - 1][c]
                    cur_sum = sub_sum[r2][c] - (sub_sum[r1 - 1][c] if r1 > 0 else 0)
                    delta = cur_sum - target
                    result += count[delta]
                    count[cur_sum] += 1
        return result
