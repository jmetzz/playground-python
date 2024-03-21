import pytest
from leetcode.p_200_number_of_islands import (
    max_island_size,
    min_island_size,
    num_islands,
    size_of_islands,
)

ZERO_ISLAND = [["0", "0", "0"], ["0", "0", "0"], ["0", "0", "0"]]
ONE_SMALL_ISLAND = [["0", "0", "0"], ["0", "1", "0"], ["0", "0", "0"]]
ONE_LARGE_ISLAND = [["1", "1", "1"], ["1", "1", "1"], ["1", "1", "1"]]
ONE_HUGE_ISLAND = [
    ["1", "1", "1", "1", "0"],
    ["1", "1", "0", "1", "0"],
    ["1", "1", "0", "0", "0"],
    ["0", "0", "0", "0", "0"],
]
TWO_ISLANDS = [["1", "1", "0"], ["0", "1", "0"], ["0", "0", "1"]]
TOW_2X2_ISLANDS = [
    ["1", "1", "0", "0", "0"],
    ["1", "1", "0", "0", "0"],
    ["0", "0", "0", "1", "1"],
    ["0", "0", "0", "1", "1"],
]
THREE_ISLANDS = [
    ["1", "1", "0", "0", "0"],
    ["1", "1", "0", "0", "0"],
    ["0", "0", "1", "0", "0"],
    ["0", "0", "0", "1", "1"],
]
FIVE_ISLANDS = [["1", "0", "1"], ["0", "1", "0"], ["1", "0", "1"]]


@pytest.mark.parametrize(
    "grid, expected",
    [
        (ZERO_ISLAND, 0),
        (ONE_SMALL_ISLAND, 1),
        (ONE_LARGE_ISLAND, 1),
        (FIVE_ISLANDS, 5),
        (TWO_ISLANDS, 2),
        (THREE_ISLANDS, 3),
    ],
)
def test_num_islands(grid, expected):
    assert num_islands(grid) == expected


@pytest.mark.parametrize(
    "grid, expected",
    [
        (ZERO_ISLAND, []),
        (ONE_LARGE_ISLAND, [9]),
        (ONE_SMALL_ISLAND, [1]),
        (FIVE_ISLANDS, [1, 1, 1, 1, 1]),
        (TOW_2X2_ISLANDS, [4, 4]),
        (ONE_HUGE_ISLAND, [9]),
        (THREE_ISLANDS, [1, 2, 4]),
    ],
)
def test_size_of_islands(grid, expected):
    assert sorted(size_of_islands(grid)) == sorted(expected)


@pytest.mark.parametrize(
    "grid, expected",
    [
        (ZERO_ISLAND, 0),
        (ONE_LARGE_ISLAND, 9),
        (ONE_SMALL_ISLAND, 1),
        (FIVE_ISLANDS, 1),
        (TOW_2X2_ISLANDS, 4),
        (ONE_HUGE_ISLAND, 9),
        (THREE_ISLANDS, 1),
    ],
)
def test_min_island_size(grid, expected):
    assert min_island_size(grid) == expected


@pytest.mark.parametrize(
    "grid, expected",
    [
        (ZERO_ISLAND, 0),
        (ONE_LARGE_ISLAND, 9),
        (ONE_SMALL_ISLAND, 1),
        (FIVE_ISLANDS, 1),
        (TOW_2X2_ISLANDS, 4),
        (ONE_HUGE_ISLAND, 9),
        (THREE_ISLANDS, 4),
    ],
)
def test_max_island_size(grid, expected):
    assert max_island_size(grid) == expected
