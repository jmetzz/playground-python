"""
1004. Max Consecutive Ones III
Medium

Given a binary array nums and an integer k, return the maximum number of
consecutive 1's in the array if you can flip at most k 0's.


Example 1:
Input: nums = [1,1,1,0,0,0,1,1,1,1,0], k = 2
Output: 6
Explanation: [1,1,1,0,0,1,1,1,1,1,1]
Bolded numbers were flipped from 0 to 1. The longest subarray is underlined.

Example 2:
Input: nums = [0,0,1,1,0,0,1,1,1,0,1,1,0,0,0,1,1,1,1], k = 3
Output: 10
Explanation: [0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1]
Bolded numbers were flipped from 0 to 1. The longest subarray is underlined.


Constraints:
1 <= nums.length <= 105
nums[i] is either 0 or 1.
0 <= k <= nums.length
"""

from typing import List


def max_len_sequence_of_ones(nums: List[int], flip_budget: int) -> int:
    """Two pointer sliding window solution.

    Rationale:
    * keep a moving window with two pointer, left and right.
    * everything in the window is considered a 1, and thus,
    the size of the window gives us the nubmer of consecutive 1s.
    * Expand the window at each iteration
    * If a zero is found, flip it if budget is still available
    * Shrink the window when a zero is found and there is no more budget to flip it.
    While shrinknig the window, there is an opportunity to claim budget back
    if a zero was dropped out of the window (the left most element was a zero).
    * Update the max number of consecutive 1s so far (take the max).
    * Continue this process until all elements in the array are processed
    """

    left, right = 0, 0
    max_ones = 0
    flipped = 0
    while right < len(nums):
        if nums[right] == 0:
            # "flip it" if a zero is found
            flipped += 1

        # should we shrink the window?
        if flipped > flip_budget:
            # shrink the window furhter if we can not
            # afford to flip more consecutive zeros
            if nums[left] == 0:
                # collect back budget:
                # if a zero element was the last
                # element of the moving window,
                # decrement the number of zeros flipped
                flipped -= 1
            left += 1

        # expands the window
        # and updates the max size of 1's sequence
        right += 1
        max_ones = max(max_ones, right - left)
    return max_ones
