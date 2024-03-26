import pytest
from leetcode.p_2360_longest_cycle_in_a_graph import (
    longest_cycle_boolean_arr,
    longest_cycle_kosaraju,
    longest_cycle_set,
    longest_cycle_side_effect,
)


@pytest.mark.parametrize(
    "func",
    [
        longest_cycle_set,
        longest_cycle_boolean_arr,
        longest_cycle_side_effect,
        longest_cycle_kosaraju,
    ],
)
@pytest.mark.parametrize("input_arr, expected", [([3, 3, 4, 2, 3], 3), ([2, -1, 3, 1], -1)])
def test_longest_cycle(func, input_arr, expected):
    assert func(input_arr[:]) == expected
