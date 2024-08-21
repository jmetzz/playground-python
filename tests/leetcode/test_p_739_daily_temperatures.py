import pytest

from leetcode.p_739_daily_temperatures import DailyTemperatures


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ([73, 74, 75, 71, 69, 72, 76, 73], [1, 1, 4, 2, 1, 1, 0, 0]),
        ([30, 40, 50, 60], [1, 1, 1, 0]),
        ([30, 60, 90], [1, 1, 0]),
    ],
)
def test_daily_temp_iterative(test_input, expected):
    assert DailyTemperatures().solve_iterative(test_input) == expected


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ([73, 74, 75, 71, 69, 72, 76, 73], [1, 1, 4, 2, 1, 1, 0, 0]),
        ([30, 40, 50, 60], [1, 1, 1, 0]),
        ([30, 60, 90], [1, 1, 0]),
    ],
)
def test_daily_temp_stack(test_input, expected):
    assert DailyTemperatures().solve_stack(test_input) == expected


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ([73, 74, 75, 71, 69, 72, 76, 73], [1, 1, 4, 2, 1, 1, 0, 0]),
        ([30, 40, 50, 60], [1, 1, 1, 0]),
        ([30, 60, 90], [1, 1, 0]),
    ],
)
def test_daily_temp_stack_2(test_input, expected):
    assert DailyTemperatures().solve_stack_2(test_input) == expected
