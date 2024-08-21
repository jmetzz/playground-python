import pytest

from leetcode.p_2073_time_needed_to_buy_tickets import (
    time_required_to_buy_1,
    time_required_to_buy_2,
    time_required_to_buy_3,
)


@pytest.mark.parametrize(
    "func",
    [time_required_to_buy_1, time_required_to_buy_2, time_required_to_buy_3],
)
@pytest.mark.parametrize(
    "input_arr, k, expected", [([2, 3, 2], 2, 6), ([5, 1, 1, 1], 0, 8), ([84, 49, 5, 24, 70, 77, 87, 8], 3, 154)]
)
def test_time_required_to_buy(func, input_arr, k, expected):
    assert func(input_arr[:], k) == expected
