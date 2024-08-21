"""https://leetcode.com/problems/equal-row-and-column-pairs/description

2352. Equal Row and Column Pairs
Medium

Given a 0-indexed n x n integer matrix grid, return the number of
pairs (ri, cj) such that row ri and column cj are equal.

A row and column pair is considered equal if they contain the same
elements in the same order (i.e., an equal array).



Example 1:
Input: grid = [[3,2,1],[1,7,6],[2,7,7]]
Output: 1
Explanation: There is 1 equal row and column pair:
- (Row 2, Column 1): [2,7,7]

Example 2:
Input: grid = [[3,1,2,2],[1,4,4,5],[2,4,2,2],[2,4,2,2]]
Output: 3
Explanation: There are 3 equal row and column pairs:
- (Row 0, Column 0): [3,1,2,2]
- (Row 2, Column 2): [2,4,2,2]
- (Row 3, Column 2): [2,4,2,2]


Constraints:
n == grid.length == grid[i].length
1 <= n <= 200
1 <= grid[i][j] <= 105
"""

from collections import Counter
from typing import List


def equal_pairs(grid: List[List[int]]) -> int:
    # Convert each row to a tuple (to make it hashable)
    # and count the occurrences using a dictionary.
    rows = Counter(tuple(row) for row in grid[:])

    # repeat the process for the colums
    # The *grid unpacks the grid, effectively passing
    # each row in grid as a separate argument to zip.
    # zip then takes the first item of each row and
    # groups them into a tuple (the first column),
    # then the second items (the second column), and so on,
    # effectively transposing the rows and columns.
    cols = Counter(zip(*grid))

    # now we can count the number of matches between rows and cols
    count = 0
    for r in rows:
        # if there is a match, increase
        # the count by the number of times
        # each row occurred in the column
        if r in cols:
            count += cols[r] * rows[r]
    return count
