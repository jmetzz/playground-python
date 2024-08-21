"""https://leetcode.com/problems/binary-subarrays-with-sum/description

930. Binary Subarrays With Sum
Medium

Given a binary array nums and an integer goal, return the number of non-empty subarrays with a sum goal.

A subarray is a contiguous part of the array.

Example 1:
Input: nums = [1,0,1,0,1], goal = 2
Output: 4
Explanation: The 4 subarrays are bolded and underlined below:
[1,0,1,0,1]
[1,0,1,0,1]
[1,0,1,0,1]
[1,0,1,0,1]

Example 2:
Input: nums = [0,0,0,0,0], goal = 0
Output: 15


Constraints:

1 <= nums.length <= 3 * 104
nums[i] is either 0 or 1.
0 <= goal <= nums.length

"""

from collections import defaultdict
from typing import List


def num_subarrays_with_sum(nums: List[int], goal: int) -> int:
    """Calculates the number of contiguous subarrays that sum to a specific goal

    This function uses prefix sums technique with a hashmap to track
    the frequency of cumulative sums (prefix sums) encountered
    while iterating through the array.
    By identifying the difference between the current cumulative sum and the goal,
    it efficiently finds subarrays that meet the goal sum without needing to
    examine each possible subarray directly.

    Intuition:
    - A running cumulative sum of elements is maintained.
    - For each new element added to the cumulative sum,
      check if the difference (current sum - goal) has been seen before.
    - If so, add the frequency of that difference to the total count,
      as it represents the number of times a subarray ending at
      the current index sums to the goal.

    Args:
    ----
        nums (List[int]): Binary array of 0s and 1s.
        goal (int): The target sum for the subarrays.

    Returns:
    -------
        int: The total number of contiguous subarrays that sum to the goal.

    Examples:
    --------
        >>> num_subarrays_with_sum([1, 0, 1, 0, 1], 2)
        4
        >>> num_subarrays_with_sum([0, 0, 0, 0, 0], 0)
        15

    """
    sum_freq = defaultdict(int)  # prefix sum -> frequency
    curr_sum = 0  # cumulative sum of elements encountered so far
    total_count = 0  # number of subarrays with the desired sum
    for e in nums:
        curr_sum += e
        if curr_sum == goal:
            total_count += 1

        # Check if there is any prefix sum that
        # can be subtracted from the current sum
        # to get the desired goal
        if curr_sum - goal in sum_freq:
            total_count += sum_freq[curr_sum - goal]

        sum_freq[curr_sum] += 1

    return total_count


def num_subarrays_with_sum_sliding_window(nums: List[int], goal: int) -> int:
    """Calculates the number of contiguous subarrays that sum to a specific goal

    by utilizing a modified sliding window approach to count subarrays
    with sums at most a target, then adjusting the count for the exact target.

    The key insight of this approach is to use the difference between
    counts of subarrays with sums at most `goal` and those with sums at most `goal - 1`
    to find the exact count of subarrays summing to `goal`.
    This about set difference.

    Intuition:
    - Define a helper function `at_most` to count subarrays with sum at most a given target.
    - The total count of subarrays with sum exactly equal to `goal` is found by
      subtracting the count of subarrays with sum at most `goal - 1` from the count with sum at most `goal`.
    - This effectively excludes subarrays whose sum exceeds the goal but includes those exactly equal.

    Args:
    ----
        nums (List[int]): Binary array of 0s and 1s.
        goal (int): The exact sum to achieve with the subarrays.

    Returns:
    -------
        int: The total number of contiguous subarrays that sum exactly to the goal.

    Examples:
    --------
        >>> num_subarrays_with_sum_sliding_window([1, 0, 1, 0, 1], 2)
        4
        >>> num_subarrays_with_sum_sliding_window([0, 0, 0, 0, 0], 0)
        15

    """

    def at_most(target: int) -> int:
        """Count the number of subarrays with sum at most the given goal"""
        start, curr_sum, answer = 0, 0, 0
        for end in range(len(nums)):
            curr_sum += nums[end]
            # shrink the window if necessary
            while start <= end and curr_sum > target:
                curr_sum -= nums[start]
                start += 1
            # The adjusted window has a total sum
            # less than or equal to goal.
            # Then, increment total count (answer)
            # by the length of the current window
            answer += end - start + 1
        return answer

    return at_most(goal) - at_most(goal - 1)


def num_subarrays_with_sum_one_pass_sliding_window(nums: List[int], goal: int) -> int:
    """Calculates the number of contiguous subarrays that sum to a specific goal

    This approaches builds on the previous and uses a one-pass sliding window iteration.
    This is achiebed by counting prefix zeros to efficiently account for additional
    valid subarrays when the goal is reached.

    Intuition:
    - The sliding window expands to include each new element and contracts when the sum exceeds the goal.
    - Zeros preceding a window that sums to the goal contribute to multiple valid configurations,
      as they can be optionally included or excluded without affecting the sum.
    - The solution iterates through the array once, maintaining a count of prefix zeros and
      adjusting the window to ensure the sum matches the goal, counting all valid configurations.

    Args:
    ----
        nums (List[int]): Binary array of 0s and 1s.
        goal (int): The target sum for the subarrays.

    Returns:
    -------
        int: The total number of contiguous subarrays summing to the goal,
            considering additional configurations provided by prefix zeros.

    Examples:
    --------
        >>> num_subarrays_with_sum_one_pass_sliding_window([1, 0, 1, 0, 1], 2)
        4
        >>> num_subarrays_with_sum_one_pass_sliding_window([0, 0, 0, 0, 0], 0)
        15

    """
    start, curr_sum, prefix_zeros, answer = 0, 0, 0, 0

    for end in range(len(nums)):
        curr_sum += nums[end]
        # shrink the window if necessary
        while start < end and (nums[start] == 0 or curr_sum > goal):
            if nums[start] == 1:
                prefix_zeros = 0
            else:
                prefix_zeros += 1
            curr_sum -= nums[start]
            start += 1
        # Count subarrays when window sum matches the goal
        if curr_sum == goal:
            answer += 1 + prefix_zeros
    return answer
