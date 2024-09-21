import pytest

from leetcode.p_130_surrounded_regions import capture_inplace


@pytest.mark.parametrize(
    "board, expected",
    [
        ([["X"]], [["X"]]),
        ([["O"]], [["O"]]),
        (
            [["X", "X", "X", "X"], ["X", "O", "O", "X"], ["X", "X", "O", "X"], ["X", "O", "X", "X"]],
            [["X", "X", "X", "X"], ["X", "X", "X", "X"], ["X", "X", "X", "X"], ["X", "O", "X", "X"]],
        ),
        (
            [
                ["X", "X", "X", "X", "X"],
                ["X", "O", "O", "O", "X"],
                ["X", "X", "O", "X", "X"],
                ["X", "O", "X", "O", "X"],
                ["X", "X", "X", "X", "X"],
            ],
            [
                ["X", "X", "X", "X", "X"],
                ["X", "X", "X", "X", "X"],
                ["X", "X", "X", "X", "X"],
                ["X", "X", "X", "X", "X"],
                ["X", "X", "X", "X", "X"],
            ],
        ),
        (
            [["O", "O", "X", "X"], ["X", "O", "X", "O"], ["X", "O", "O", "X"], ["X", "X", "X", "X"]],
            [["O", "O", "X", "X"], ["X", "O", "X", "O"], ["X", "O", "O", "X"], ["X", "X", "X", "X"]],
        ),
        (
            [
                ["O", "X", "O", "X", "O"],
                ["X", "O", "O", "X", "X"],
                ["X", "X", "O", "O", "X"],
                ["X", "O", "X", "X", "O"],
                ["O", "X", "X", "O", "O"],
            ],
            [
                ["O", "X", "O", "X", "O"],
                ["X", "O", "O", "X", "X"],
                ["X", "X", "O", "O", "X"],
                ["X", "X", "X", "X", "O"],
                ["O", "X", "X", "O", "O"],
            ],
        ),
    ],
)
def test_capture_region(board, expected):
    capture_inplace(board)
    assert board == expected
