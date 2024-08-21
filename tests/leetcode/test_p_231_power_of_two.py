import pytest

from leetcode.p_231_power_of_two import (
    power_of_two__solve_bitwise,
    power_of_two__solve_bitwise2,
    power_of_two__solve_loop,
    power_of_two__solve_loop_2,
    power_of_two__solve_mathematically,
)


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (1, True),
        (16, True),
        (3, False),
        (536870912, True),
    ],
)
def test_is_power_of_two_mathematically(test_input, expected):
    assert power_of_two__solve_mathematically(test_input) == expected


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (1, True),
        (16, True),
        (3, False),
        (536870912, True),
    ],
)
def test_is_power_of_two_loop(test_input, expected):
    assert power_of_two__solve_loop(test_input) == expected


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (1, True),
        (16, True),
        (3, False),
        (536870912, True),
    ],
)
def test_is_power_of_two_loop2(test_input, expected):
    assert power_of_two__solve_loop_2(test_input) == expected


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (1, True),
        (16, True),
        (3, False),
        (536870912, True),
    ],
)
def test_is_power_of_two_bitwise(test_input, expected):
    assert power_of_two__solve_bitwise(test_input) == expected


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (1, True),
        (16, True),
        (3, False),
        (536870912, True),
    ],
)
def test_is_power_of_two_bitwise2(test_input, expected):
    assert power_of_two__solve_bitwise2(test_input) == expected
