"""https://leetcode.com/problems/number-of-islands/description/

200. Number of Islands
Medium

Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water),
return the number of islands.

An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically.
You may assume all four edges of the grid are all surrounded by water.


Example 1:
Input: grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
Output: 1

Example 2:
Input: grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
Output: 3

Constraints:

m == grid.length
n == grid[i].length
1 <= m, n <= 300
grid[i][j] is '0' or '1'.
"""

from math import inf
from typing import List


def num_islands(grid: List[List[str]]) -> int:
    """Traverse without side-effects on the grid object.

    If allowed to change the input grid, the visited is not necessary,
    since we can simulate it by chaning "land" to "water", ie, 1 -> 0.
    """
    ROWS = len(grid)
    COLUMNS = len(grid[0])
    visited = set()

    def _traverse(r: int, c: int) -> int:
        # check for position boundary
        # - is row inbounds?
        # - is col inbounds?
        if not ((0 <= r < ROWS) and (0 <= c < COLUMNS)):
            return False
        if (r, c) in visited or grid[r][c] == "0":
            return False

        # to avoid the visited set (saving memory),
        # replace then next line by grid[r][c] = "0"
        # and remove all references to visited
        visited.add((r, c))

        # traverse the grid as far as possible
        # marking all land cells as visited
        _traverse(r - 1, c)  # upper cell
        _traverse(r + 1, c)  # lower cell
        _traverse(r, c - 1)  # left cell
        _traverse(r, c + 1)  # right cell
        return True

    count = 0
    for row in range(ROWS):
        for col in range(COLUMNS):
            if _traverse(row, col) is True:
                count += 1

    return count


def size_of_islands(grid: List[List[str]]) -> List[int]:
    ROWS = len(grid)
    COLUMNS = len(grid[0])
    visited = set()

    def _traverse(r: int, c: int) -> int:
        # check for position boundary
        # - is row inbounds?
        # - is col inbounds?
        if not ((0 <= r < ROWS) and (0 <= c < COLUMNS)):
            return 0
        # check if can proceed with this position
        if (r, c) in visited or grid[r][c] == "0":
            return 0
        visited.add((r, c))
        size = 1  # count the current position
        size += _traverse(r - 1, c)  # upper cell
        size += _traverse(r + 1, c)  # lower cell
        size += _traverse(r, c - 1)  # left cell
        size += _traverse(r, c + 1)  # right cell

        return size

    islands_size = []
    for row in range(ROWS):
        for col in range(COLUMNS):
            size = _traverse(row, col)
            if size > 0:
                islands_size.append(size)
    return sorted(islands_size)


def min_island_size(grid: List[List[str]]) -> int:
    ROWS = len(grid)
    COLUMNS = len(grid[0])
    visited = set()

    def _traverse(r: int, c: int) -> int:
        # check for position boundary
        # - is row inbounds?
        # - is col inbounds?
        if not (0 <= r < ROWS and 0 <= c < COLUMNS):
            return 0

        if (r, c) in visited or grid[r][c] == "0":
            return 0

        visited.add((r, c))
        size = 1  # accounts for the current land cell
        size += _traverse(r - 1, c)
        size += _traverse(r + 1, c)
        size += _traverse(r, c - 1)
        size += _traverse(r, c + 1)
        return size

    min_size = inf
    for row in range(ROWS):
        for col in range(COLUMNS):
            curr_size = _traverse(row, col)
            if curr_size > 0 and curr_size < min_size:
                min_size = curr_size
    return min_size if min_size != inf else 0


def max_island_size(grid: List[List[str]]) -> int:
    ROWS = len(grid)
    COLUMNS = len(grid[0])
    visited = set()

    def _traverse(r: int, c: int) -> int:
        # check for position boundary
        # - is row inbounds?
        # - is col inbounds?
        if not (0 <= r < ROWS and 0 <= c < COLUMNS):
            return 0

        if (r, c) in visited or grid[r][c] == "0":
            return 0

        visited.add((r, c))
        size = 1  # accounts for the current land cell
        size += _traverse(r - 1, c)
        size += _traverse(r + 1, c)
        size += _traverse(r, c - 1)
        size += _traverse(r, c + 1)
        return size

    max_size = -inf
    for row in range(ROWS):
        for col in range(COLUMNS):
            curr_size = _traverse(row, col)
            max_size = max(curr_size, max_size)
    return max_size


if __name__ == "__main__":
    grid = [["1", "1", "1", "1", "0"], ["1", "1", "0", "1", "0"], ["1", "1", "0", "0", "0"], ["0", "0", "0", "0", "0"]]
    print(f"#num islands: {num_islands(grid)}, with sizes {size_of_islands(grid)}")
    print(f"#smalles island: {min_island_size(grid)}")
    print(f"#biggest island: {max_island_size(grid)}")

    grid = [["1", "1", "0", "0", "0"], ["1", "1", "0", "0", "0"], ["0", "0", "1", "0", "0"], ["0", "0", "0", "1", "1"]]
    print(f"#num islands: {num_islands(grid)}, with sizes {size_of_islands(grid)}")
    print(f"#smalles island: {min_island_size(grid)}")
    print(f"#biggest island: {max_island_size(grid)}")

    grid = [["0", "0", "0", "0"], ["0", "0", "0", "0"], ["0", "0", "0", "0"]]
    print(f"#num islands: {num_islands(grid)}, with sizes {size_of_islands(grid)}")
    print(f"#smalles island: {min_island_size(grid)}")
    print(f"#biggest island: {max_island_size(grid)}")
