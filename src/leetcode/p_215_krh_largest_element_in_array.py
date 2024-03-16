"""
https://leetcode.com/problems/course-schedule/description/

215. Kth Largest Element in an Array
Medium

Given an integer array nums and an integer k, return the kth largest element in the array.

Note that it is the kth largest element in the sorted order, not the kth distinct element.

Can you solve it without sorting?


Example 1:
Input: nums = [3,2,1,5,6,4], k = 2
Output: 5

Example 2:
Input: nums = [3,2,3,1,2,4,5,5,6], k = 4
Output: 4


Constraints:
1 <= k <= nums.length <= 105
-104 <= nums[i] <= 104
"""

from heapq import heapify, heappop
from typing import List


def find_kth_largest_heap(nums: List[int], k: int) -> int:
    arr = [-e for e in nums]
    heapify(arr)
    for _ in range(k - 1):
        heappop(arr)
    return -arr[0]


def find_kth_largest_counting(nums: List[int], k: int) -> int:
    min_val, max_val = min(nums), max(nums)
    counter = [0] * (max_val - min_val + 1)
    for num in nums:
        counter[num - min_val] += 1

    remain = k
    for num in range(len(counter) - 1, -1, -1):
        remain -= counter[num]
        if remain <= 0:
            return num + min_val

    return -1


if __name__ == "__main__":
    inputs = [([3, 2, 3, 1, 2, 4, 5, 5, 6], 2), ([3, 2, 1, 5, 6, 4], 4)]
    for arr, k in inputs:
        print(find_kth_largest_counting(arr, k))
        print(find_kth_largest_heap(arr, k))
        print("---")
