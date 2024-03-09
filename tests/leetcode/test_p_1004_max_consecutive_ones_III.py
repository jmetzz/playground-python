import pytest
from leetcode.p_1004_max_consecutive_ones_III import max_len_sequence_of_ones


@pytest.mark.parametrize(
    "input_arr, flip_budget, expected",
    [
        ([1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0], 2, 6),
        ([0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1], 3, 10),
        ([1, 1], 0, 2),
        ([0, 0, 0, 0], 1, 1),
        ([0, 0, 0, 0], 0, 0),
    ],
)
def test_max_len_sequence_of_ones(input_arr, flip_budget, expected):
    assert max_len_sequence_of_ones(input_arr, flip_budget) == expected
