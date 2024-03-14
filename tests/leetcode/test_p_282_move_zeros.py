import pytest
from leetcode.p_282_move_zeros import move_zeroes_fast_slow, move_zeroes_loops

test_cases = [
    ([0, 1, 0, 3, 12], [1, 3, 12, 0, 0]),  # Basic functionality
    ([0, 0, 0, 0], [0, 0, 0, 0]),  # All zeros
    ([1, 2, 3], [1, 2, 3]),  # No zeros
    ([0], [0]),  # Single zero
    ([1], [1]),  # Single non-zero
    ([], []),  # Empty list
]


@pytest.mark.parametrize("input_list, expected", test_cases)
def test_move_zeroes_loops(input_list, expected):
    move_zeroes_loops(input_list)
    assert input_list == expected, f"Expected {expected}, got {input_list} after move_zeroes_loops"


@pytest.mark.parametrize("input_list, expected", test_cases)
def test_move_zeroes_fast_slow(input_list, expected):
    move_zeroes_fast_slow(input_list)
    assert input_list == expected, f"Expected {expected}, got {input_list} after move_zeroes_fast_slow"
