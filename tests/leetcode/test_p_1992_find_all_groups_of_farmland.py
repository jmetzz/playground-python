import pytest
from leetcode.p_1992_find_all_groups_of_farmland import (
    find_farmland_iterative,
    find_farmland_recursive_1,
    find_farmland_recursive_2,
)


@pytest.mark.parametrize("func", [find_farmland_recursive_1, find_farmland_recursive_2, find_farmland_iterative])
@pytest.mark.parametrize(
    "input_grid, expected",
    [
        ([], []),
        ([[0]], []),
        ([[1]], [[0, 0, 0, 0]]),
        ([[1, 1], [1, 1]], [[0, 0, 1, 1]]),
        ([[1, 0, 0], [0, 1, 1], [0, 1, 1]], [[0, 0, 0, 0], [1, 1, 2, 2]]),
    ],
)
def test_find_all_groups_of_farmland(func, input_grid, expected):
    assert func(input_grid) == expected
