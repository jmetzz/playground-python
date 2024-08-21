import pytest

from leetcode.p_45_jump_game_II import jump_greedy


@pytest.mark.parametrize("input_data, expected", [([2, 3, 1, 1, 4], 2), ([2, 3, 0, 1, 4], 2)])
def test_jump_II(input_data, expected):
    assert jump_greedy(input_data) == expected
