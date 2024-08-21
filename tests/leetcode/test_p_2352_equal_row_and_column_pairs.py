import pytest

from leetcode.p_2352_equal_row_and_column_pairs import equal_pairs


@pytest.mark.parametrize(
    "input_grid, expected",
    [
        ([[3, 2, 1], [1, 7, 6], [2, 7, 7]], 1),
        ([[3, 1, 2, 2], [1, 4, 4, 5], [2, 4, 2, 2], [2, 4, 2, 2]], 3),
        ([[13, 13], [13, 13]], 4),
    ],
)
def test_equal_pairs(input_grid, expected):
    assert equal_pairs(input_grid) == expected
