from typing import List

import pytest
from hypothesis import given
from hypothesis.strategies import integers, lists

from data_structures.matrices import max_columns_width, serialize, transpose


def test_transpose_square_matrix():
    actual = transpose([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert actual == [[1, 4, 7], [2, 5, 8], [3, 6, 9]], "Should transpose a 3x3 matrix correctly"


def test_transpose_single_element_matrix():
    assert transpose([[1]]) == [[1]], "Should handle single element matrix"


def test_transpose_empty_matrix():
    assert transpose([]) == [], "Should handle an empty matrix"


def test_transpose_non_square_matrix():
    with pytest.raises(ValueError) as exc_info:
        transpose([[1, 2, 3], [4, 5, 6]], square=True)
    expected_msg = "Expect a quadratic matrix (n==n). Given 2x3"
    assert str(exc_info.value) == expected_msg, "Should raise ValueError for non-square matrices"


def test_transpose_two_by_two_matrix():
    assert transpose([[1, 2], [3, 4]]) == [[1, 3], [2, 4]], "Should transpose a 2x2 matrix correctly"


@given(
    integers(min_value=1, max_value=5).flatmap(
        lambda n: lists(lists(integers(), min_size=n, max_size=n), min_size=n, max_size=n)
    )
)
def test_transpose_twice(matrix: List[List[int]]):
    """Test that transposing a square matrix twice returns the original matrix."""
    # Ensure the matrix is square for the test
    assert len(matrix) == len(matrix[0]), "Matrix is not square"
    assert transpose(transpose(matrix)) == matrix, "Double transposition should return the original matrix"


@pytest.mark.parametrize(
    "input_matrix, expected",
    [
        (None, []),
        ([[]], []),
        ([[1]], [1]),
        ([[1, 2, 3]], [1, 1, 1]),
        ([[1], [2], [3]], [1]),
        ([[1, 2, 3], [4, 5, 6]], [1, 1, 1]),
        ([[1, 2, 3], [4, 50, 6], [4, 5, 600]], [1, 2, 3]),
    ],
)
def test_max_columns_width(input_matrix, expected):
    assert max_columns_width(input_matrix) == expected


@pytest.mark.parametrize(
    "input_matrix, expected",
    [
        (None, ""),
        ([[]], "|  |"),
        ([[1]], "| 1 |"),
        ([[1, 2, 3]], "| 1 | 2 | 3 |"),
        ([[1], [2], [3]], "| 1 |\n| 2 |\n| 3 |"),
        (
            [[1, 2, 3], [4, 5, 6]],
            "| 1 | 2 | 3 |\n| 4 | 5 | 6 |",
        ),
        (
            [[1, 2, 3], [4, 50, 6], [4, 5, 600]],
            "| 1 |  2 |   3 |\n| 4 | 50 |   6 |\n| 4 |  5 | 600 |",
        ),
    ],
)
def test_matrices_justified(input_matrix, expected):
    assert serialize(input_matrix, justify=True) == expected
