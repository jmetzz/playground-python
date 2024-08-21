import pytest

from leetcode.p_498_diagonal_traverse import (
    find_diagonal_natural_order,
    find_diagonal_order_simple,
    find_diagonal_order_zigzag,
)

test_data_zigzag_order = [
    # Simple 3x3 matrix
    ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [1, 2, 4, 7, 5, 3, 6, 8, 9]),
    # 2x2 matrix
    ([[1, 2], [3, 4]], [1, 2, 3, 4]),
    # 2x3 matrix
    ([[1, 2, 3], [4, 5, 6]], [1, 2, 4, 5, 3, 6]),
    # 3x2 matrix
    ([[1, 2], [3, 4], [5, 6]], [1, 2, 3, 5, 4, 6]),
    # Single element
    ([[1]], [1]),
    # Single row
    ([[1, 2, 3]], [1, 2, 3]),
    # Single column
    ([[1], [2], [3]], [1, 2, 3]),
    # Matrix with negative and positive integers
    ([[-105, 1], [1, 105]], [-105, 1, 1, 105]),
]


@pytest.mark.parametrize("func", [find_diagonal_order_zigzag, find_diagonal_order_simple])
@pytest.mark.parametrize("matrix, expected_output", test_data_zigzag_order)
def test_find_diagonal_zigzag_order(func, matrix, expected_output):
    assert func(matrix) == expected_output, f"Failed for matrix: {matrix}"


test_data_natural_order = [
    # Simple 3x3 matrix
    ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [1, 2, 4, 3, 5, 7, 6, 8, 9]),
    # 2x2 matrix
    ([[1, 2], [3, 4]], [1, 2, 3, 4]),
    # 2x3 matrix
    ([[1, 2, 3], [4, 5, 6]], [1, 2, 4, 3, 5, 6]),
    # 3x2 matrix
    ([[1, 2], [3, 4], [5, 6]], [1, 2, 3, 4, 5, 6]),
    # Single element
    ([[1]], [1]),
    # Single row
    ([[1, 2, 3]], [1, 2, 3]),
    # Single column
    ([[1], [2], [3]], [1, 2, 3]),
    # Matrix with negative and positive integers
    ([[-105, 1], [1, 105]], [-105, 1, 1, 105]),
]


@pytest.mark.parametrize("matrix, expected_output", test_data_natural_order)
def test_find_diagonal_order_natural(matrix, expected_output):
    assert find_diagonal_natural_order(matrix) == expected_output, f"Failed for matrix: {matrix}"
