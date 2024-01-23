import pytest

from conftest import HUGE_MATRIX
from leetcode.p_931_min_falling_path_sum import MinFallingPathSumViaRecursion, MinFallingPathSumViaDP


@pytest.mark.parametrize("test_input, expected",
                         [
                             ([[2, 1, 3], [6, 5, 4], [7, 8, 9]], 13),
                             ([[-19, 57], [-40, -5]], -59),
                             ([[-48]], -48),
                             (HUGE_MATRIX, 10_000)
                         ])
def test_min_falling_path_sum_via_loop(test_input, expected):
    assert expected == MinFallingPathSumViaRecursion().solve(test_input)


@pytest.mark.parametrize("test_input, expected",
                         [
                             ([[2, 1, 3], [6, 5, 4], [7, 8, 9]], 13),
                             ([[-19, 57], [-40, -5]], -59),
                             ([[-48]], -48),
                             (HUGE_MATRIX, 10_000)
                         ])
def test_min_falling_path_sum_via_loop(test_input, expected):
    assert expected == MinFallingPathSumViaDP().solve(test_input)
