import pytest

from leetcode.p_1732_find_the_highest_altitude import highest_altitude_1, highest_altitude_2


@pytest.mark.parametrize("func", [highest_altitude_1, highest_altitude_2])
@pytest.mark.parametrize("input_arr, expected", [([-5, 1, 5, 0, -7], 1), ([-4, -3, -2, -1, 4, 3, 2], 0)])
def test_highest_altitude(func, input_arr, expected):
    assert func(input_arr) == expected
