import pytest

from leetcode.p_3005_count_elements_with_max_frequency import (
    max_frequency_elements_double_loop,
    max_frequency_elements_single_loop,
)


@pytest.mark.parametrize("input_arr, expected", [([1, 2, 2, 3, 1, 4], 4), ([1, 2, 3, 4, 5], 5)])
def test_max_frequency_elements_double_loop(input_arr, expected):
    assert max_frequency_elements_double_loop(input_arr) == expected


@pytest.mark.parametrize("input_arr, expected", [([1, 2, 2, 3, 1, 4], 4), ([1, 2, 3, 4, 5], 5)])
def test_max_frequency_elements_single_loop(input_arr, expected):
    assert max_frequency_elements_single_loop(input_arr) == expected
