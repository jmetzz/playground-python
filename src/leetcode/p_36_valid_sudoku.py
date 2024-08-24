"""
https://leetcode.com/problems/valid-sudoku/description

36. Valid Sudoku
Medium

Determine if a 9 x 9 Sudoku board is valid.
Only the filled cells need to be validated according to the following rules:
- Each row must contain the digits 1-9 without repetition.
- Each column must contain the digits 1-9 without repetition.
- Each of the nine 3 x 3 sub-boxes of the grid must contain the digits 1-9 without repetition.

Note:
- A Sudoku board (partially filled) could be valid but is not necessarily solvable.
- Only the filled cells need to be validated according to the mentioned rules.


Example 1:
Input: board =
[["5","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]]
Output: true

Example 2:
Input: board =
[["8","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]]
Output: false
Explanation: Same as Example 1, except with the 5 in the top left corner being modified to 8.
Since there are two 8's in the top left 3x3 sub-box, it is invalid.


Constraints:
board.length == 9
board[i].length == 9
board[i][j] is a digit 1-9 or '.'
"""

from collections import defaultdict


def is_valid_sudoku(board: list[list[str]], hidden_char: str = ".") -> bool:
    _line_map = defaultdict(set)
    _col_map = defaultdict(set)
    _block_map = defaultdict(set)

    for row in range(9):
        for col in range(9):
            value = board[row][col]
            if value == hidden_char:
                continue
            # curr block is defined as tuple(start_row, start_col)
            curr_block = ((row // 3) * 3, (col // 3) * 3)

            if value in _line_map[row] or value in _col_map[col] or value in _block_map[curr_block]:
                return False

            _line_map[row].add(value)
            _col_map[col].add(value)
            _block_map[curr_block].add(value)
    return True


def get_sudoku_sub_matrix(sudoku, row, col):
    # Determine the starting row and column of the sub-matrix
    start_row = (row // 3) * 3
    start_col = (col // 3) * 3

    # Extract the 3x3 sub-matrix
    return [sudoku[i][start_col : start_col + 3] for i in range(start_row, start_row + 3)]


if __name__ == "__main__":
    board = [
        ["5", "3", ".", ".", "7", ".", ".", ".", "."],
        ["6", ".", ".", "1", "9", "5", ".", ".", "."],
        [".", "9", "8", ".", ".", ".", ".", "6", "."],
        ["8", ".", ".", ".", "6", ".", ".", ".", "3"],
        ["4", ".", ".", "8", ".", "3", ".", ".", "1"],
        ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
        [".", "6", ".", ".", ".", ".", "2", "8", "."],
        [".", ".", ".", "4", "1", "9", ".", ".", "5"],
        [".", ".", ".", ".", "8", ".", ".", "7", "9"],
    ]
    print(is_valid_sudoku(board))  # true

    board = [
        ["8", "3", ".", ".", "7", ".", ".", ".", "."],
        ["6", ".", ".", "1", "9", "5", ".", ".", "."],
        [".", "9", "8", ".", ".", ".", ".", "6", "."],
        ["8", ".", ".", ".", "6", ".", ".", ".", "3"],
        ["4", ".", ".", "8", ".", "3", ".", ".", "1"],
        ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
        [".", "6", ".", ".", ".", ".", "2", "8", "."],
        [".", ".", ".", "4", "1", "9", ".", ".", "5"],
        [".", ".", ".", ".", "8", ".", ".", "7", "9"],
    ]
    print(is_valid_sudoku(board))  # false

    board = [
        [".", ".", "4", ".", ".", ".", "6", "3", "."],
        [".", ".", ".", ".", ".", ".", ".", ".", "."],
        ["5", ".", ".", ".", ".", ".", ".", "9", "."],
        [".", ".", ".", "5", "6", ".", ".", ".", "."],
        ["4", ".", "3", ".", ".", ".", ".", ".", "1"],
        [".", ".", ".", "7", ".", ".", ".", ".", "."],
        [".", ".", ".", "5", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", ".", "."],
    ]
    print(is_valid_sudoku(board))  # false; repeat 5 in column 3

    board = [  #
        [".", ".", ".", ".", ".", ".", "5", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", ".", "."],
        ["9", "3", ".", ".", "2", ".", "4", ".", "."],
        [".", ".", "7", ".", ".", ".", "3", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", "3", "4", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", "3", ".", ".", "."],
        [".", ".", ".", ".", ".", "5", "2", ".", "."],
    ]
    print(is_valid_sudoku(board))  # false; repeat 3 in block 8

    sub_matrix = get_sudoku_sub_matrix(board, 6, 4)
    for row in sub_matrix:
        print(row)
