import pytest
from leetcode.p_268_missing_number import (
    missing_number_solve_naive_loops,
    missing_number_solve_set,
    missing_number_solve_sorting,
    missing_number_solve_sum,
    missing_number_solve_xor,
)


@pytest.mark.parametrize(
    "test_input, expected",
    [([3, 0, 1], 2), ([0, 1], 2), ([9, 6, 4, 2, 3, 5, 7, 0, 1], 8)],
)
def test_missing_number_solve_set(test_input, expected):
    assert missing_number_solve_set(test_input) == expected


@pytest.mark.parametrize(
    "test_input, expected",
    [([3, 0, 1], 2), ([0, 1], 2), ([9, 6, 4, 2, 3, 5, 7, 0, 1], 8)],
)
def test_missing_number_solve_loops(test_input, expected):
    assert missing_number_solve_naive_loops(test_input) == expected


@pytest.mark.parametrize(
    "test_input, expected",
    [([3, 0, 1], 2), ([0, 1], 2), ([9, 6, 4, 2, 3, 5, 7, 0, 1], 8)],
)
def test_missing_number_xor(test_input, expected):
    assert missing_number_solve_xor(test_input) == expected


@pytest.mark.parametrize(
    "test_input, expected",
    [([3, 0, 1], 2), ([0, 1], 2), ([9, 6, 4, 2, 3, 5, 7, 0, 1], 8)],
)
def test_missing_number_sum(test_input, expected):
    assert missing_number_solve_sum(test_input) == expected


@pytest.mark.parametrize(
    "test_input, expected",
    [([3, 0, 1], 2), ([0, 1], 2), ([9, 6, 4, 2, 3, 5, 7, 0, 1], 8)],
)
def test_missing_number_sorting(test_input, expected):
    assert missing_number_solve_sorting(test_input) == expected
