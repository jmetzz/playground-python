import pytest

from leetcode.p_54_spiral_matrix import spiral_order


@pytest.mark.parametrize(
    "input_matrix, expected",
    [
        ([[1]], [1]),
        ([[1, 2, 3]], [1, 2, 3]),
        ([[1], [2], [3]], [1, 2, 3]),
        ([[1, 2], [4, 3]], [1, 2, 3, 4]),
        ([[1, 2, 3], [8, 9, 4], [7, 6, 5]], [1, 2, 3, 4, 5, 6, 7, 8, 9]),
        ([[1, 2, 3, 4], [10, 11, 12, 5], [9, 8, 7, 6]], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
        ([[1, 2, 3], [6, 5, 4]], [1, 2, 3, 4, 5, 6]),
        (
            [[1, 2, 3, 4], [12, 13, 14, 5], [11, 16, 15, 6], [10, 9, 8, 7]],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        ),
        ([[-1, -2, -3], [-4, -5, -6], [-7, -8, -9]], [-1, -2, -3, -6, -9, -8, -7, -4, -5]),
    ],
)
def test_spiral_order(input_matrix, expected):
    assert spiral_order(input_matrix) == expected
