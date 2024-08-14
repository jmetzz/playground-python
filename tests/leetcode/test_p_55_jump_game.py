import pytest
from leetcode.p_55_jump_game import can_jump_greedy, can_jump_loop, can_jump_recursion


@pytest.mark.parametrize("func", [can_jump_recursion, can_jump_loop, can_jump_greedy])
@pytest.mark.parametrize("input_data, expected", [([2, 3, 1, 1, 4], True), ([3, 2, 1, 0, 4], False)])
def test_function_name(func, input_data, expected):
    assert func(input_data) == expected
