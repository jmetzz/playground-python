"""
https://leetcode.com/problems/minimum-common-value/description/
2540. Minimum Common Value
Easy

Given two integer arrays nums1 and nums2, sorted in non-decreasing order,
return the minimum integer common to both arrays.
If there is no common integer amongst nums1 and nums2, return -1.

Note that an integer is said to be common to nums1 and nums2
if both arrays have at least one occurrence of that integer.

Example 1:
Input: nums1 = [1,2,3], nums2 = [2,4]
Output: 2
Explanation: The smallest element common to both arrays is 2, so we return 2.

Example 2:
Input: nums1 = [1,2,3,6], nums2 = [2,3,4,5]
Output: 2
Explanation: There are two common elements in the array 2 and 3 out of which
2 is the smallest, so 2 is returned.

Constraints:
1 <= nums1.length, nums2.length <= 105
1 <= nums1[i], nums2[j] <= 109
Both nums1 and nums2 are sorted in non-decreasing order.
"""

import bisect
from typing import List


def get_common__with_set_intersection(nums1: List[int], nums2: List[int]) -> int:
    """This approach uses set intersection and sorting"""
    intersection = sorted(set(nums1) & set(nums2))
    if intersection:
        return intersection[0]
    return -1


def get_common__with_loop_and_set(nums1: List[int], nums2: List[int]) -> int:
    """This approach searches for elements from the shorter array in the longer array set of elements"""
    shorter, longer = nums1, nums2
    if len(longer) < len(shorter):
        shorter, longer = longer, shorter
    longer_set = set(longer)
    for e in shorter:
        if e in longer_set:
            return e
    return -1


def get_common__with_bin_search(nums1: List[int], nums2: List[int]) -> int:
    """This approache use binary search"""
    shorter, longer = nums1, nums2
    if len(longer) < len(shorter):
        shorter, longer = longer, shorter

    for e in shorter:
        idx = bisect.bisect_left(longer, e)
        if longer[idx] == e:
            return e
    return -1


def get_common__with_two_pointers(nums1: List[int], nums2: List[int]) -> int:
    """This approach uses two pointers"""
    i, j = 0, 0
    while i < len(nums1) and j < len(nums2):
        if nums1[i] == nums2[j]:
            return nums1[i]
        if nums1[i] < nums2[j]:
            i += 1
        else:
            j += 1
    return -1
