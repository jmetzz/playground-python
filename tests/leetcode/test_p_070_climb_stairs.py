import pytest

from leetcode.p_070_climb_stairs import (
    climb_stairs_memoization,
    climb_stairs_space_optimization,
    climb_stairs_tabulation,
)


@pytest.mark.parametrize("test_input, expected", [(0, 1), (1, 1), (2, 2), (3, 3), (4, 5)])
def test_climb_stairs_memoization(test_input, expected):
    assert climb_stairs_memoization(test_input) == expected


@pytest.mark.parametrize("test_input, expected", [(0, 1), (1, 1), (2, 2), (3, 3), (4, 5)])
def test_climb_stairs_tabulation(test_input, expected):
    assert climb_stairs_tabulation(test_input) == expected


@pytest.mark.parametrize("test_input, expected", [(0, 1), (1, 1), (2, 2), (3, 3), (4, 5)])
def test_climb_stairs_space_optimization(test_input, expected):
    assert climb_stairs_space_optimization(test_input) == expected
