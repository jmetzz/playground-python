"""
https://leetcode.com/problems/count-subarrays-where-max-element-appears-at-least-k-times

2962. Count Subarrays Where Max Element Appears at Least K Times
Medium

You are given an integer array nums and a positive integer k.

Return the number of subarrays where the maximum element of nums appears at least k times in that subarray.

A subarray is a contiguous sequence of elements within an array.


Example 1:
Input: nums = [1,3,2,3,3], k = 2
Output: 6
Explanation: The subarrays that contain the element 3 at least 2 times are:
[1,3,2,3], [1,3,2,3,3], [3,2,3], [3,2,3,3], [2,3,3] and [3,3].

Example 2:
Input: nums = [1,4,2,1], k = 3
Output: 0
Explanation: No subarray contains the element 4 at least 3 times.


Constraints:

1 <= nums.length <= 105
1 <= nums[i] <= 106
1 <= k <= 105
"""

from typing import List


def count_subarrays(nums: List[int], k: int) -> int:
    """
    Counts the number of subarrays in which the maximum element appears exactly k times.

    This function works by sliding a window across the input list `nums`. For each new element
    added to the window, it checks if this element equals the maximum element of the entire list.
    If so, it increases the frequency count of the maximum element within the current window.

    The window expands until the frequency of the maximum element reaches `k`. At this point, the
    window starts to shrink from the left, reducing the frequency count of the maximum element
    if necessary, until the frequency is less than `k` again.

    The intuition behind accumulating the counter is based on the observation that for a window
    ending at the current element and having the maximum element appearing exactly k times,
    there are as many valid subarrays ending at this position as the start index of the window.
    This is because any subarray starting from the beginning of the array up to the start of
    the window does not affect the condition of having the maximum element appearing exactly
    k times within the window.

    Args:
        nums (List[int]): The list of integers to process.
        k (int): The exact number of times the maximum element must appear in a subarray for it to be counted.

    Returns:
        int: The count of subarrays meeting the specified condition.

    Example:
        >>> count_subarrays([1, 4, 2, 4, 3, 4], 2)
        9
        This output means there are 9 subarrays where the maximum element (4 in this case) appears exactly 2 times.
    """

    max_element = max(nums)
    counter, start, window_freq_of_max = 0, 0, 0

    for e in nums:
        if e == max_element:
            window_freq_of_max += 1

        while window_freq_of_max == k:
            if nums[start] == max_element:
                window_freq_of_max -= 1
            start += 1
        counter += start
    return counter
