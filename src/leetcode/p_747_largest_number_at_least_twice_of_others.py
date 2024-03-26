"""
747. Largest Number At Least Twice of Others

https://leetcode.com/problems/largest-number-at-least-twice-of-others/description/
Easy

You are given an integer array nums where the largest integer is unique.

Determine whether the largest element in the array is at least twice as much as every other number in the array.
If it is, return the index of the largest element, or return -1 otherwise.



Example 1:
Input: nums = [3,6,1,0]
Output: 1
Explanation: 6 is the largest integer.
For every other number in the array x, 6 is at least twice as big as x.
The index of value 6 is 1, so we return 1.

Example 2:
Input: nums = [1,2,3,4]
Output: -1
Explanation: 4 is less than twice the value of 3, so we return -1.


Constraints:

2 <= nums.length <= 50
0 <= nums[i] <= 100
The largest element in nums is unique.


Hint:
Scan through the array to find the unique largest element `m`,
keeping track of it's index `maxIndex`. Scan through the array again.
If we find some `x != m` with `m < 2*x`, we should return `-1`.
Otherwise, we should return `maxIndex`.
"""

from heapq import heapify
from typing import List


def dominant_index(nums: List[int]) -> int:
    max_idx = 0
    for idx in range(1, len(nums)):
        if nums[idx] > nums[max_idx]:
            max_idx = idx
    for i, e in enumerate(nums):
        if i == max_idx:
            continue
        if e * 2 > nums[max_idx]:
            return -1
    return max_idx


def dominant_index_heap(nums: List[int]) -> int:
    if len(nums) == 1:
        return 0

    h = [(-v, idx) for idx, v in enumerate(nums)]
    heapify(h)
    largest_v, largest_idx = h[0]
    sec_largest_v, _ = min(h[1 : min(3, len(nums))], key=lambda x: x[0])

    if sec_largest_v * 2 < largest_v:
        return -1
    return largest_idx


if __name__ == "__main__":
    print(dominant_index([3, 0]))  # expect 0
    print(dominant_index([0, 3]))  # expect 1
    print(dominant_index([3, 6, 1, 0]))  # expect 1
    print(dominant_index([1, 2, 3, 4]))  # expect -1
    print()
    print(dominant_index_heap([3, 0]))  # expect 0
    print(dominant_index_heap([0, 3]))  # expect 1
    print(dominant_index_heap([3, 6, 1, 0]))  # expect 1
    print(dominant_index_heap([1, 2, 3, 4]))  # expect -1
