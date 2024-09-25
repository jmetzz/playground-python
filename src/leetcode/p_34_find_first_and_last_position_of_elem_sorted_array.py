"""https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/description/

34. Find First and Last Position of Element in Sorted Array
Medium

Given an array of integers nums sorted in non-decreasing order,
find the starting and ending position of a given target value.

If target is not found in the array, return [-1, -1].

You must write an algorithm with O(log n) runtime complexity.


Example 1:
Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]

Example 2:
Input: nums = [5,7,7,8,8,10], target = 6
Output: [-1,-1]

Example 3:
Input: nums = [], target = 0
Output: [-1,-1]


Constraints:
0 <= nums.length <= 105
-109 <= nums[i] <= 109
nums is a non-decreasing array.
-109 <= target <= 109
"""

from bisect import bisect_left, bisect_right


def search_range_loop(nums: list[int], target: int) -> list[int]:
    if not nums or nums[0] > target or nums[-1] < target:
        return [-1, -1]
    for i in range(len(nums)):
        if nums[i] == target:
            start = i
            while i + 1 < len(nums) and nums[i + 1] == target:
                i += 1
            return [start, i]
    return [-1, -1]


def search_range_bisect(nums: list[int], target: int) -> list[int]:
    if not nums or nums[0] > target or nums[-1] < target:
        return [-1, -1]

    left = bisect_left(nums, target)
    if left == len(nums) or nums[left] != target:
        return [-1, -1]

    right = bisect_right(nums, target)
    # Since bisect_right returns the index after the last occurrence of target,
    # we do not need to check if nums[right - 1] == target.
    return [left, right - 1]


def search_range_find(nums: list[int], target: int) -> list[int]:
    if not nums or nums[0] > target or nums[-1] < target:
        return [-1, -1]
    try:
        left = nums.index(target)
    except ValueError:
        return [-1, -1]

    right = nums[::-1].index(target)
    return [left, len(nums) - right - 1]


def search_range_bin_search(nums: list[int], target: int) -> list[int]:
    def _find_left_index(lo, hi):
        if nums[lo] == target:
            # base case for arrays of len 1
            return lo
        while lo <= hi:
            mid = (lo + hi) // 2
            if nums[mid] == target and nums[mid - 1] < target:
                return mid
            if nums[mid] < target:
                lo = mid + 1
            else:
                hi = mid - 1
        return -1

    def _find_right_index(lo, hi):
        if nums[hi] == target:
            # base case for arrays of len 1
            return hi
        while lo <= hi:
            mid = (lo + hi) // 2
            if nums[mid] == target and nums[mid + 1] > target:
                return mid
            if nums[mid] > target:
                hi = mid - 1
            else:
                lo = mid + 1
        return -1

    if not nums or nums[0] > target or nums[-1] < target:
        return [-1, -1]
    left = _find_left_index(0, len(nums) - 1)
    right = _find_right_index(0, len(nums) - 1)
    return [left, right]


def search_range_bin_search_2(nums: list[int], target: int) -> list[int]:
    def _bin_search(lo, hi, target):
        while lo < hi:
            mid = lo + (hi - lo) // 2
            if nums[mid] >= target:
                hi = mid
            else:
                lo = mid + 1
        return lo

    if not nums or nums[0] > target or nums[-1] < target:
        return [-1, -1]
    left = _bin_search(0, len(nums) - 1, target)
    if nums[left] != target:
        return [-1, -1]
    right = _bin_search(0, len(nums) - 1, target + 1)
    return [left, right] if nums[right] == target else [left, right - 1]
