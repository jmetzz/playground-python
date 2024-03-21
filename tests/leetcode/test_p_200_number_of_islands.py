import pytest
from leetcode.p_200_number_of_islands import num_islands, size_of_islands  # Adjust the import path as needed


@pytest.mark.parametrize(
    "grid, expected",
    [
        ([["0", "0", "0"], ["0", "0", "0"], ["0", "0", "0"]], 0),  # No islands
        ([["1", "1", "1"], ["1", "1", "1"], ["1", "1", "1"]], 1),  # Single large island
        ([["0", "0", "0"], ["0", "1", "0"], ["0", "0", "0"]], 1),  # Single small island
        ([["1", "0", "1"], ["0", "1", "0"], ["1", "0", "1"]], 5),  # Multiple islands
        ([["1", "1", "0"], ["0", "1", "0"], ["0", "0", "1"]], 2),  # Multiple islands
        (
            [
                ["1", "1", "0", "0", "0"],
                ["1", "1", "0", "0", "0"],
                ["0", "0", "1", "0", "0"],
                ["0", "0", "0", "1", "1"],
            ],
            3,
        ),
    ],
)
def test_num_islands(grid, expected):
    assert num_islands(grid) == expected


@pytest.mark.parametrize(
    "grid, expected",
    [
        ([["0", "0", "0"], ["0", "0", "0"], ["0", "0", "0"]], []),  # No islands
        ([["1", "1", "1"], ["1", "1", "1"], ["1", "1", "1"]], [9]),  # Single large island
        ([["0", "0", "0"], ["0", "1", "0"], ["0", "0", "0"]], [1]),  # Single small island
        ([["1", "0", "1"], ["0", "1", "0"], ["1", "0", "1"]], [1, 1, 1, 1, 1]),  # Multiple separate islands
        (
            [
                ["1", "1", "0", "0", "0"],
                ["1", "1", "0", "0", "0"],
                ["0", "0", "0", "1", "1"],
                ["0", "0", "0", "1", "1"],
            ],
            [4, 4],
        ),
        (
            [
                ["1", "1", "1", "1", "0"],
                ["1", "1", "0", "1", "0"],
                ["1", "1", "0", "0", "0"],
                ["0", "0", "0", "0", "0"],
            ],
            [9],
        ),
        (
            [
                ["1", "1", "0", "0", "0"],
                ["1", "1", "0", "0", "0"],
                ["0", "0", "1", "0", "0"],
                ["0", "0", "0", "1", "1"],
            ],
            [1, 2, 4],
        ),
    ],
)
def test_size_of_islands(grid, expected):
    assert sorted(size_of_islands(grid)) == sorted(expected)
