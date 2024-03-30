"""
https://leetcode.com/problems/subarrays-with-k-different-integers/description

992. Subarrays with K Different Integers
Hard

Given an integer array nums and an integer k, return the number of good subarrays of nums.

A good array is an array where the number of different integers in that array is exactly k.

For example, [1,2,3,1,2] has 3 different integers: 1, 2, and 3.
A subarray is a contiguous part of an array.



Example 1:
Input: nums = [1,2,1,2,3], k = 2
Output: 7
Explanation: Subarrays formed with exactly 2 different integers:
[1,2], [2,1], [1,2], [2,3], [1,2,1], [2,1,2], [1,2,1,2]

Example 2:
Input: nums = [1,2,1,3,4], k = 3

Output: 3
Explanation: Subarrays formed with exactly 3 different integers:
[1,2,1,3], [2,1,3], [1,3,4].


Constraints:

1 <= nums.length <= 2 * 104
1 <= nums[i], k <= nums.length
"""

from collections import defaultdict
from typing import List


def subarrays_with_k_distinct(nums: List[int], k: int) -> int:
    """
    One pass expand the shrink a sliding-window

    The logic updates the sliding window accoding to the conditions:
        * expand: add the current num
        * shrink: until the window contains exactly k distinct elements

    The important aspect to solve this problem is:
        the subarrays with more than k distinct integers are irrelevant, and thus,
        we should only consider subarrays with k or less than k distinct integers.

    The calculation right - left + 1, counts the subarrays with **at most** k distinct integers.
    Thus, it is still necessary to isolate the subarrays that **strictly meet the target k**.

    """
    window_nums = defaultdict(int)

    total_counter = 0
    curr_count = 0
    lo = 0
    for num in nums:
        # The for-loop controls the expansion of the window.
        # We only need to ensure we considering the number
        # in the window_nums counter
        window_nums[num] += 1

        # Decrement k everytime a new element enters the window,
        # effectivelly using k to controle the number of distinct elements in the window
        if window_nums[num] == 1:
            k -= 1

        if k < 0:
            # If k becomes negative, it means we added too many distinct elements to the window.
            # Therefore, we need to shrink the window from the left
            # until the count of distinct elements becomes valid again (k == 0)
            dropped = nums[lo]  # get the element that will drop from the window
            window_nums[dropped] -= 1  # decrement its frequency in the reminder of the window
            if window_nums[dropped] == 0:
                # by dropping the element that had only 1 occurante in the window
                # it means we have one less distinct element in the window,
                # represented by increasing k
                k += 1

            lo += 1  # advance the left bounday of the window (effectivelly droping the left most element)
            curr_count = 0

        if k == 0:
            # Anytime k equals zero we have a good window,
            # and thus, we can calculate the number of good subarrays.
            # While the count of left remains greater than 1,
            # keep shrinking the window from the left
            while window_nums[nums[lo]] > 1:
                window_nums[nums[lo]] -= 1
                lo += 1
                curr_count += 1
            # Add the count of subarrays with K distinct elements to the total count
            total_counter += curr_count + 1

    return total_counter
