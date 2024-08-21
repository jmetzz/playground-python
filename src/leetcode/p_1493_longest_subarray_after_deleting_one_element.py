"""https://leetcode.com/problems/longest-subarray-of-1s-after-deleting-one-element/description

1493. Longest Subarray of 1's After Deleting One Element
Medium
Topics
Companies
Hint
Given a binary array nums, you should delete one element from it.

Return the size of the longest non-empty subarray containing only 1's in the resulting array.
Return 0 if there is no such subarray.


Example 1:
Input: nums = [1,1,0,1]
Output: 3
Explanation: After deleting the number in position 2, [1,1,1] contains 3 numbers with value of 1's.

Example 2:
Input: nums = [0,1,1,1,0,1,1,0,1]
Output: 5
Explanation: After deleting the number in position 4,
[0,1,1,1,1,1,0,1] longest subarray with value of 1's is [1,1,1,1,1].

Example 3:
Input: nums = [1,1,1]
Output: 2
Explanation: You must delete one element.


Constraints:

1 <= nums.length <= 105
nums[i] is either 0 or 1.
"""

from typing import List


def longest_subarray_sliding_window(nums: List[int]) -> int:
    zero_counter, longest_window = 0, 0
    lo = 0
    for idx, num in enumerate(nums):
        if num == 0:
            zero_counter += 1

        while zero_counter > 1:
            if nums[lo] == 0:
                zero_counter -= 1
            lo += 1

        window_size = idx - lo  # discounting 1 zero
        # This strategy covers the edge case with no zeroes,
        # since the zero_counter never exceeds 1 and the window
        # will cover the whole array.
        # In the end, the difference between the first and last index
        # would provide the array size minus 1, which is intended
        # as we need to delete one element.
        longest_window = max(window_size, longest_window)

    return longest_window
