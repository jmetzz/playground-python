"""https://leetcode.com/problems/word-search/description

79. Word Search
Medium

Given an m x n grid of characters board and a string word, return true if word exists in the grid.

The word can be constructed from letters of sequentially adjacent cells,
where adjacent cells are horizontally or vertically neighboring.
The same letter cell may not be used more than once.



Example 1:
Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
Output: true

Example 2:
Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "SEE"
Output: true

Example 3:
Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCB"
Output: false


Constraints:

m == board.length
n = board[i].length
1 <= m, n <= 6
1 <= word.length <= 15
board and word consists of only lowercase and uppercase English letters.
"""


def exist_1(board: list[list[str]], word: str, debug: bool = False) -> bool:
    n = len(board)
    m = len(board[0])

    if len(word) > n * m:
        return False

    def dfs_explore_from(r, c, idx) -> bool:
        if not (0 <= r < n and 0 <= c < m):
            return False
        if (r, c) in used:
            if debug:
                print(f"used: ({r}, {c})")
            return False
        if debug:
            print(f"({r}, {c}) : {board[r][c]} | {word[idx:]}")
        if idx == len(word) - 1:
            return word[idx] == board[r][c]

        if idx < len(word) and word[idx] != board[r][c]:
            return False

        used.add((r, c))
        combined = (
            dfs_explore_from(r, c + 1, idx + 1)  # right
            or dfs_explore_from(r + 1, c, idx + 1)  # down
            or dfs_explore_from(r, c - 1, idx + 1)  # left
            or dfs_explore_from(r - 1, c, idx + 1)  # up
        )
        used.remove((r, c))
        return combined

    used = set()
    for row in range(n):
        for col in range(m):
            used.clear()
            if dfs_explore_from(row, col, 0) is True:
                return True
    return False


def exist_2(board: list[list[str]], word: str, debug: bool = False) -> bool:
    n = len(board)
    m = len(board[0])

    if len(word) > n * m:
        return False

    def dfs_explore_from(r, c, idx) -> bool:
        if idx == len(word):
            return True

        if not (0 <= r < n and 0 <= c < m) or word[idx] != board[r][c]:
            return False

        curr_char = board[r][c]
        board[r][c] = ""  # mark it as used
        answer = dfs_explore_from(r, c + 1, idx + 1)  # right
        answer |= dfs_explore_from(r + 1, c, idx + 1)  # down
        answer |= dfs_explore_from(r, c - 1, idx + 1)  # left
        answer |= dfs_explore_from(r - 1, c, idx + 1)  # up
        board[r][c] = curr_char  # reinstante the character
        return answer

    for row in range(n):
        for col in range(m):
            if dfs_explore_from(row, col, 0) is True:
                return True
    return False


def exist_3(board: list[list[str]], word: str, debug: bool = False) -> bool:
    n = len(board)
    m = len(board[0])

    if len(word) > n * m:
        return False

    def dfs_explore_from(r, c, idx) -> bool:
        if idx == len(word):
            return True

        if not (0 <= r < n and 0 <= c < m) or word[idx] != board[r][c]:
            return False

        curr_char = board[r][c]
        board[r][c] = ""  # mark it as used
        # use short circuit to speed up the process
        answer = (
            dfs_explore_from(r, c + 1, idx + 1)  # right
            or dfs_explore_from(r + 1, c, idx + 1)  # down
            or dfs_explore_from(r, c - 1, idx + 1)  # left
            or dfs_explore_from(r - 1, c, idx + 1)  # up
        )
        board[r][c] = curr_char  # always reinstante the character to avoid side-effects
        return answer

    for row in range(n):
        for col in range(m):
            if dfs_explore_from(row, col, 0) is True:
                return True
    return False
