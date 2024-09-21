"""
https://leetcode.com/problems/surrounded-regions/description

130. Surrounded Regions

Medium

You are given an m x n matrix board containing letters 'X' and 'O',
capture regions that are surrounded:

* Connect: A cell is connected to adjacent cells horizontally or vertically.
* Region: To form a region connect every 'O' cell.
* Surround: The region is surrounded with 'X' cells if you can connect the region with 'X' cells
and none of the region cells are on the edge of the board.

A surrounded region is captured by replacing all 'O's with 'X's in the input matrix board.



Example 1:

Input: board = [["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]

Output: [["X","X","X","X"],["X","X","X","X"],["X","X","X","X"],["X","O","X","X"]]

Explanation:


In the above diagram, the bottom region is not captured because it is on the
edge of the board and cannot be surrounded.

Example 2:

Input: board = [["X"]]

Output: [["X"]]



Constraints:

m == board.length
n == board[i].length
1 <= m, n <= 200
board[i][j] is 'X' or 'O'.

"""

from data_structures.matrices import serialize


def capture_inplace(board: list[list[int]], captured: str = "X", target: str = "O") -> None:
    """
    Do not return anything, modify board in-place instead.
    """
    ROWS, COLS = len(board), len(board[0])
    mask = "T"

    def _dfs_capture(r, c, mask_value):
        if r < 0 or r == ROWS or c < 0 or c == COLS or board[r][c] != target:
            # out of boundary or not target
            return

        board[r][c] = mask_value
        _dfs_capture(r + 1, c, mask_value)
        _dfs_capture(r - 1, c, mask_value)
        _dfs_capture(r, c - 1, mask_value)
        _dfs_capture(r, c + 1, mask_value)

    # phase 1: identify the unsurrounded regions - map 'target' to 'mark'
    for row in range(ROWS):
        for col in range(COLS):
            if board[row][col] == target and row in (0, ROWS - 1) or col in (0, COLS - 1):
                # for every border target cell, mark as "non-capturable"
                _dfs_capture(row, col, mask)

    # phase 2: capture all surrounded regions
    for row in range(ROWS):
        for col in range(COLS):
            if board[row][col] == target:
                board[row][col] = captured
            elif board[row][col] == mask:
                board[row][col] = target


if __name__ == "__main__":
    board = [["X", "X", "X", "X"], ["X", "O", "O", "X"], ["X", "X", "O", "X"], ["X", "O", "X", "X"]]
    capture_inplace(board)
    print(serialize(board))
    print("-" * 20)

    board = [["X"]]
    capture_inplace(board)
    print(serialize(board))
    print("-" * 20)

    board = [
        ["X", "X", "X", "X", "X"],
        ["X", "O", "O", "O", "X"],
        ["X", "X", "O", "X", "X"],
        ["X", "O", "X", "O", "X"],
        ["X", "X", "X", "X", "X"],
    ]
    capture_inplace(board)
    print(serialize(board))
    print("-" * 20)

    board = [["O", "O", "X", "X"], ["X", "O", "X", "O"], ["X", "O", "O", "X"], ["X", "X", "X", "X"]]
    capture_inplace(board)
    print(serialize(board))
    print("-" * 20)

    board = [
        ["O", "X", "O", "X", "O"],
        ["X", "O", "O", "X", "X"],
        ["X", "X", "O", "O", "X"],
        ["X", "O", "X", "X", "O"],
        ["O", "X", "X", "O", "O"],
    ]
    capture_inplace(board)
    print(serialize(board))
