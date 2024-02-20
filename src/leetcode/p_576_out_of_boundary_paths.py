from typing import Dict


class OutOfBoundaryPaths:
    def solve_recursive(
        self,
        num_rows: int,
        num_cols: int,
        max_moves: int,
        start_row: int,
        start_col: int,
    ) -> int:
        if not (0 <= start_row < num_rows and 0 <= start_col < num_cols):
            return 1

        if max_moves <= 0:
            return 0

        counter = (
            self.solve_recursive(num_rows, num_cols, max_moves - 1, start_row - 1, start_col)  # up
            + self.solve_recursive(num_rows, num_cols, max_moves - 1, start_row + 1, start_col)  # down
            + self.solve_recursive(num_rows, num_cols, max_moves - 1, start_row, start_col - 1)  # left
            + self.solve_recursive(num_rows, num_cols, max_moves - 1, start_row, start_col + 1)  # right
        )

        return counter % 1000000007  # 10**9 + 7

    def solve_memo(
        self,
        num_rows: int,
        num_cols: int,
        max_moves: int,
        start_row: int,
        start_col: int,
        cache: Dict,
    ) -> int:
        if (start_row, start_col, max_moves) in cache:
            return cache[(start_row, start_col, max_moves)]

        if not (0 <= start_row < num_rows and 0 <= start_col < num_cols):
            return 1

        if max_moves <= 0:
            return 0

        counter = (
            self.solve_memo(num_rows, num_cols, max_moves - 1, start_row - 1, start_col, cache)  # up
            + self.solve_memo(num_rows, num_cols, max_moves - 1, start_row + 1, start_col, cache)  # down
            + self.solve_memo(num_rows, num_cols, max_moves - 1, start_row, start_col - 1, cache)  # left
            + self.solve_memo(num_rows, num_cols, max_moves - 1, start_row, start_col + 1, cache)  # right
        ) % 1000000007  # module 10**9 + 7
        cache[(start_row, start_col, max_moves)] = counter

        return counter
