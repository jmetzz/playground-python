"""
https://leetcode.com/problems/game-of-life/description/

289. Game of Life
Medium

According to Wikipedia's article: "The Game of Life, also known simply as Life,
is a cellular automaton devised by the British mathematician
John Horton Conway in 1970."

The board is made up of an m x n grid of cells, where each cell has
an initial state: live (represented by a 1) or dead (represented by a 0).
Each cell interacts with its eight neighbors (horizontal, vertical, diagonal)
using the following four rules (taken from the above Wikipedia article):

1. Any live cell with fewer than two live neighbors dies as if
caused by under-population.
2. Any live cell with two or three live neighbors lives on to the next generation.
3. Any live cell with more than three live neighbors dies, as if by over-population.
4. Any dead cell with exactly three live neighbors becomes a live cell,
as if by reproduction.

The next state is created by applying the above rules simultaneously to
every cell in the current state, where births and deaths occur simultaneously.
Given the current state of the m x n grid board, return the next state.

Example 1:
Input: board = [[0,1,0],[0,0,1],[1,1,1],[0,0,0]]
Output: [[0,0,0],[1,0,1],[0,1,1],[0,1,0]]

Example 2:
Input: board = [[1,1],[1,0]]
Output: [[1,1],[1,1]]

Constraints:

m == board.length
n == board[i].length
1 <= m, n <= 25
board[i][j] is 0 or 1.

Follow up:
Could you solve it in-place? Remember that the board needs to be updated
simultaneously: You cannot update some cells first and then use their
updated values to update other cells.
In this question, we represent the board using a 2D array.
In principle, the board is infinite, which would cause problems when
the active area encroaches upon the border of the array
(i.e., live cells reach the border). How would you address these problems?
"""

from data_structures.matrices import serialize


def game_of_life_naive(board: list[list[int]]) -> None:
    """
    Do not return anything, modify board in-place instead.
    """

    def _count_live_neighbors(row, col):
        _live_count = 0
        for delta_row, delta_col in NEIGHBORS:
            r, c = row + delta_row, col + delta_col
            if 0 <= r < ROWS and 0 <= c < COLS:
                _live_count += board[r][c]
        return _live_count

    NEIGHBORS = (
        (-1, -1),
        (-1, 0),
        (-1, 1),  # Top-left, Top, Top-right
        (0, -1),
        (0, 1),  # Left, Right
        (1, -1),
        (1, 0),
        (1, 1),  # Bottom-left, Bottom, Bottom-right
    )
    ROWS, COLS = len(board), len(board[0])
    new_board = [[0 for _ in range(COLS)] for _ in range(ROWS)]
    for row in range(ROWS):
        for col in range(COLS):
            live_neighbors = _count_live_neighbors(row, col)

            if board[row][col] == 1:
                if live_neighbors < 2 or live_neighbors > 3:
                    # The cell dies: under-population or over-population
                    new_board[row][col] = 0
                else:
                    # The cell lives
                    new_board[row][col] = 1
            else:
                if live_neighbors == 3:
                    # A dead cell with exactly 3
                    # live neighbors becomes a live cell
                    new_board[row][col] = 1

    for row in range(ROWS):
        for col in range(COLS):
            board[row][col] = new_board[row][col]


def game_of_life_optmized(board: list[list[int]]) -> None:
    def _count_live_neighbors(row, col):
        _live_count = 0
        for delta_row, delta_col in NEIGHBORS:
            r, c = row + delta_row, col + delta_col
            if 0 <= r < ROWS and 0 <= c < COLS:
                _live_count += 1 if board[r][c] in (1, 3) else 0
        return _live_count

    NEIGHBORS = (
        (-1, -1),
        (-1, 0),
        (-1, 1),  # Top-left, Top, Top-right
        (0, -1),
        (0, 1),  # Left, Right
        (1, -1),
        (1, 0),
        (1, 1),  # Bottom-left, Bottom, Bottom-right
    )
    ROWS, COLS = len(board), len(board[0])

    # stage 1, then stage 2 mapping
    # for each element that represent the current state,
    # we map to a value according to this logic:
    #   - define the stage 1 value by applying the 4 given rules
    #     (see problem description).
    #
    #   current state | stage 1 (apply rules) | stage 2
    #   ---------------------------------------------------
    #             0   | 0, if it should be 0  |   0 - dead
    #             1   | 1, if it should be 0  |   0 - dead
    #             0   | 2, if it should be 1  |   1 - alive
    #             1   | 3, if it should be 1  |   1 - alive

    for row in range(ROWS):
        for col in range(COLS):
            live_neighbors = _count_live_neighbors(row, col)
            if board[row][col]:
                # current state: alive
                if live_neighbors in (2, 3):
                    board[row][col] = 3
            elif live_neighbors == 3:
                # current state: dead
                board[row][col] = 2

    # stage 2
    for row in range(ROWS):
        for col in range(COLS):
            if board[row][col] == 1:
                board[row][col] = 0
            elif board[row][col] in (2, 3):
                board[row][col] = 1


if __name__ == "__main__":
    grid = [[0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 0, 0]]
    game_of_life_naive(grid)
    print(serialize(grid))
    print()

    grid = [[0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 0, 0]]
    game_of_life_optmized(grid)
    print(serialize(grid))
