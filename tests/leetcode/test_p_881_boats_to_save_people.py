import pytest

from leetcode.p_881_boats_to_save_people import num_rescue_boats, num_rescue_boats__inplace


@pytest.mark.parametrize("func", [num_rescue_boats, num_rescue_boats__inplace])
@pytest.mark.parametrize(
    "input_arr, limit, expected", [([3, 2, 2, 1], 3, 3), ([2, 1], 3, 1), ([3, 5, 3, 4], 5, 4), ([2, 2], 6, 1)]
)
def test_num_rescue_boats(func, input_arr, limit, expected):
    assert func(input_arr[:], limit) == expected
