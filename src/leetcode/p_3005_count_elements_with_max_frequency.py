"""
https://leetcode.com/problems/count-elements-with-maximum-frequency/description/
3005. Count Elements With Maximum Frequency
Easy

You are given an array nums consisting of positive integers.

Return the total frequencies of elements in nums such that those elements
all have the maximum frequency.

The frequency of an element is the number of occurrences of that element in the array.



Example 1:

Input: nums = [1,2,2,3,1,4]
Output: 4
Explanation: The elements 1 and 2 have a frequency of 2 which is the maximum frequency in the array.
So the number of elements in the array with maximum frequency is 4.

Example 2:
Input: nums = [1,2,3,4,5]
Output: 5
Explanation: All elements of the array have a frequency of 1 which is the maximum.
So the number of elements in the array with maximum frequency is 5.


Constraints:
1 <= nums.length <= 100
1 <= nums[i] <= 100

"""

from typing import List


def max_frequency_elements_double_loop(nums: List[int]) -> int:
    # since 1 <= nums[i] <= 100, we need [0, 100] indices
    # position zero will be ignored later
    histogram = [0] * 101
    max_freq = 0
    for e in nums:
        histogram[e] += 1
        max_freq = max(max_freq, histogram[e])
    answer = 0
    for idx in range(100):
        if histogram[idx + 1] == max_freq:
            answer += histogram[idx + 1]
    return answer


def max_frequency_elements_single_loop(nums: List[int]) -> int:
    # since 1 <= nums[i] <= 100, we need [0, 100] indices
    # position zero will be ignored later
    histogram = [0] * 101
    max_freq = 0
    answer = 0
    for e in nums:
        histogram[e] += 1
        if histogram[e] > max_freq:
            # new max_freq found -> reset counter
            max_freq = histogram[e]
            answer = max_freq
        elif histogram[e] == max_freq:
            answer += max_freq
    return answer
