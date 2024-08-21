import pytest

from leetcode.p_930_binary_subarrays_with_sum import (
    num_subarrays_with_sum,
    num_subarrays_with_sum_one_pass_sliding_window,
    num_subarrays_with_sum_sliding_window,
)

test_cases = [
    ([], 0, 0),  # Empty array, expecting 0 subarrays
    ([1], 1, 1),  # Single element in array equals the goal
    ([1], 2, 0),  # Single element in array cannot sum to the goal
    ([0], 2, 0),  # Single element in array cannot sum to the goal
    ([0], 0, 1),  # Single element in array cannot sum to the goal
    ([1, 0, 1, 1, 0], 2, 5),  # Multiple non-overlapping subarrays meet the goal
    ([1, 0, 1, 0, 1], 2, 4),  # Multiple non-overlapping subarrays meet the goal
    ([0, 0, 0, 0, 0], 0, 15),  # All zeros, goal is 0
    ([0, 0, 0, 0, 1], 1, 5),  # Only one subarray equals the goal (multiple times due to zeros)
    ([1, 1, 1], 2, 2),  # Overlapping subarrays meet the goal
    ([0, 1, 0, 1], 1, 6),  # Goal achieved with subarrays containing a single 1
    ([1, 1, 0, 1, 1], 3, 2),  # Multiple overlapping subarrays meet the goal
]


@pytest.mark.parametrize("nums, goal, expected", test_cases)
def test_num_subarrays_with_sum(nums, goal, expected):
    assert num_subarrays_with_sum(nums, goal) == expected


@pytest.mark.parametrize("nums, goal, expected", test_cases)
def test_num_subarrays_with_sum_sliding_window(nums, goal, expected):
    assert num_subarrays_with_sum_sliding_window(nums, goal) == expected


@pytest.mark.parametrize("nums, goal, expected", test_cases)
def test_num_subarrays_with_sum_one_pass_sliding_window(nums, goal, expected):
    assert num_subarrays_with_sum_one_pass_sliding_window(nums, goal) == expected
