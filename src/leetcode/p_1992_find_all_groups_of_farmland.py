"""
https://leetcode.com/problems/find-all-groups-of-farmland/description

1992. Find All Groups of Farmland
Medium

You are given a 0-indexed m x n binary matrix land where a 0 represents a hectare of forested land
and a 1 represents a hectare of farmland.

To keep the land organized, there are designated rectangular areas of hectares that
consist entirely of farmland. These rectangular areas are called groups.
No two groups are adjacent, meaning farmland in one group is not four-directionally
adjacent to another farmland in a different group.

land can be represented by a coordinate system where the top left corner of land is (0, 0)
and the bottom right corner of land is (m-1, n-1).

Find the coordinates of the top left and bottom right corner of each group of farmland.
A group of farmland with a top left corner at (r1, c1) and a bottom right corner at (r2, c2)
is represented by the 4-length array [r1, c1, r2, c2].

Return a 2D array containing the 4-length arrays described above for each group
of farmland in land. If there are no groups of farmland, return an empty array.

You may return the answer in any order.



Example 1:
Input: land = [[1,0,0],[0,1,1],[0,1,1]]
Output: [[0,0,0,0],[1,1,2,2]]
Explanation:
The first group has a top left corner at land[0][0] and a bottom right corner at land[0][0].
The second group has a top left corner at land[1][1] and a bottom right corner at land[2][2].

Example 2:
Input: land = [[1,1],[1,1]]
Output: [[0,0,1,1]]
Explanation:
The first group has a top left corner at land[0][0] and a bottom right corner at land[1][1].

Example 3:
Input: land = [[0]]
Output: []
Explanation:
There are no groups of farmland.


Constraints:

m == land.length
n == land[i].length
1 <= m, n <= 300
land consists of only 0's and 1's.
Groups of farmland are rectangular in shape.

"""

from typing import List


def find_farmland_recursive_1(land: List[List[int]]) -> List[List[int]]:
    if not land:
        return []

    def _explore(row, col, curr_group):
        if (row, col) in visited:
            return

        row_in_bounds = 0 <= row < rows
        col_in_bounds = 0 <= col < cols
        if not row_in_bounds or not col_in_bounds or land[row][col] == 0:
            return

        visited.add((row, col))
        curr_group.append([row, col])
        # even though land groups are rectangular, we still need to
        # explore all directions to ensure we mark the entire group
        # as visited
        _explore(row, col + 1, curr_group)  # explore right
        _explore(row, col - 1, curr_group)  # explore left
        _explore(row + 1, col, curr_group)  # explore down
        _explore(row - 1, col, curr_group)  # explore up

    answer = []
    visited = set()
    rows = len(land)
    cols = len(land[0])
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and land[r][c] == 1:
                curr_group = []  # starting cell
                _explore(r, c, curr_group)
                curr_group.sort()
                answer.append(curr_group[0] + curr_group[-1])

    return answer


def find_farmland_recursive_2(land: List[List[int]]) -> List[List[int]]:
    if not land:
        return []

    def _explore(row, col):
        if (row, col) in visited:
            return

        row_in_bounds = 0 <= row < rows
        col_in_bounds = 0 <= col < cols
        if not row_in_bounds or not col_in_bounds or land[row][col] == 0:
            return
        visited.add((row, col))

        nonlocal right_corner
        right_corner = max(right_corner, [row, col])

        # even though land groups are rectangular, we still need to
        # explore all directions to ensure we mark the entire group
        # as visited
        _explore(row, col + 1)  # explore right
        _explore(row, col - 1)  # explore left
        _explore(row + 1, col)  # explore down
        _explore(row - 1, col)  # explore up

    answer = []
    visited = set()
    rows = len(land)
    cols = len(land[0])
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and land[r][c] == 1:
                left_corner = [r, c]  # starting cell
                right_corner = [0, 0]
                _explore(r, c)
                answer.append(left_corner + right_corner)

    return answer


def find_farmland_iterative(land: List[List[int]]) -> List[List[int]]:
    if not land:
        return []

    rows = len(land)
    cols = len(land[0])
    answer = []

    visited = set()
    for r in range(rows):
        for c in range(cols):
            if (r, c) in visited or land[r][c] == 0:
                continue

            botton_row = r
            while botton_row + 1 < rows and land[botton_row + 1][c] == 1:
                botton_row += 1

            right_col = c
            while right_col + 1 < cols and land[r][right_col + 1] == 1:
                right_col += 1

            # mark the group land cell as visited.
            # Here we use a set to avoid side-effects on the input matrix.
            # If side-effects are not a problem, then we could instead
            # change the value of the cell inplace to mark it as visited.
            # for example, land[i2][j2] = 2
            for v_row in range(r, botton_row + 1):
                for v_col in range(c, right_col + 1):
                    visited.add((v_row, v_col))
            answer.append([r, c, botton_row, right_col])
    return answer


if __name__ == "__main__":
    print(find_farmland_recursive_2([[1, 0, 0], [0, 1, 1], [0, 1, 1]]))
