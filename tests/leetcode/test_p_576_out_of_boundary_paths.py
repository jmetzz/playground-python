import pytest
from leetcode.p_576_out_of_boundary_paths import OutOfBoundaryPaths


@pytest.mark.parametrize(
    "m, n, max_moves, start_row, start_col, expected",
    [
        (2, 2, 0, 0, 0, 0),
        (2, 2, 1, 0, 0, 2),
        (2, 2, 2, 0, 0, 6),
        (1, 3, 3, 0, 1, 12),
    ],
)
def test_out_of_boundary_paths_recursive(m, n, max_moves, start_row, start_col, expected):
    actual = OutOfBoundaryPaths().solve_recursive(m, n, max_moves, start_row, start_col)
    assert actual == expected


@pytest.mark.parametrize(
    "m, n, max_moves, start_row, start_col, expected",
    [
        (2, 2, 0, 0, 0, 0),
        (2, 2, 1, 0, 0, 2),
        (2, 2, 2, 0, 0, 6),
        (1, 3, 3, 0, 1, 12),
        (8, 7, 16, 1, 5, 102984580),
    ],
)
def test_out_of_boundary_paths_memo(m, n, max_moves, start_row, start_col, expected):
    actual = OutOfBoundaryPaths().solve_memo(m, n, max_moves, start_row, start_col, {})
    assert actual == expected
