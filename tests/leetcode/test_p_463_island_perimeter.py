import pytest

from leetcode.p_463_island_perimeter import island_perimeter_iterative

test_cases = [
    ([[0, 1, 0, 0], [1, 1, 1, 0], [0, 1, 0, 0], [1, 1, 0, 0]], 16),
    ([[1]], 4),
    ([[1, 0]], 4),
    ([[1, 1, 1], [1, 0, 1]], 12),
]


@pytest.mark.parametrize("input_grid, expected", test_cases)
def test_island_perimeter(input_grid, expected):
    assert island_perimeter_iterative(input_grid) == expected
