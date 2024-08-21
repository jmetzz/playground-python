import pytest

from leetcode.p_84_largest_rectangle_in_histogram import largest_rectangle_area


@pytest.mark.parametrize("input_arr, expected", [([2, 1, 5, 6, 2, 3], 10), ([2, 4], 4)])
def test_largest_rectangle_area(input_arr, expected):
    assert largest_rectangle_area(input_arr) == expected
