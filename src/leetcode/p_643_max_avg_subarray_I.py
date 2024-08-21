"""https://leetcode.com/problems/maximum-average-subarray-i/description

643. Maximum Average Subarray I
Easy
Topics
Companies
You are given an integer array nums consisting of n elements, and an integer k.

Find a contiguous subarray whose length is equal to k that has the maximum
average value and return this value. Any answer with a calculation error less than 10-5 will be accepted.



Example 1:

Input: nums = [1,12,-5,-6,50,3], k = 4
Output: 12.75000
Explanation: Maximum average is (12 - 5 - 6 + 50) / 4 = 51 / 4 = 12.75
Example 2:

Input: nums = [5], k = 1
Output: 5.00000


Constraints:

n == nums.length
1 <= k <= n <= 10**5
-104 <= nums[i] <= 10**4
"""

from typing import List


def find_max_average_1(nums: List[int], k: int) -> float:
    window_sum = sum(nums[:k])
    max_avg = window_sum / k

    for start in range(len(nums) - k):
        window_sum = window_sum - nums[start] + nums[start + k]
        curr_avg = window_sum / k
        max_avg = max(curr_avg, max_avg)
    return max_avg


def find_max_average_2(nums: List[int], k: int) -> float:
    window_sum = sum(nums[:k])
    max_avg = window_sum / k
    start = 0
    for end in range(k, len(nums)):
        window_sum -= nums[start]
        window_sum += nums[end]
        start += 1
        curr_avg = window_sum / k
        max_avg = max(curr_avg, max_avg)
    return max_avg
