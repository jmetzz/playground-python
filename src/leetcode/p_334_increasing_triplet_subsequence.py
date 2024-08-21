"""https://leetcode.com/problems/increasing-triplet-subsequence/description/

334. Increasing Triplet Subsequence
Medium

Given an integer array nums, return true if there exists a triple of indices (i, j, k)
such that i < j < k and nums[i] < nums[j] < nums[k]. If no such indices exists, return false.



Example 1:

Input: nums = [1,2,3,4,5]
Output: true
Explanation: Any triplet where i < j < k is valid.

Example 2:
Input: nums = [5,4,3,2,1]
Output: false
Explanation: No triplet exists.

Example 3:
Input: nums = [2,1,5,0,4,6]
Output: true
Explanation: The triplet (3, 4, 5) is valid because nums[3] == 0 < nums[4] == 4 < nums[5] == 6.


Constraints:
1 <= nums.length <= 5 * 105
-231 <= nums[i] <= 231 - 1

Follow up: Could you implement a solution that runs in O(n) time complexity and O(1) space complexity?
"""

from typing import List


def increasing_triplet(nums: List[int]) -> bool:
    """ "
    Use two variables, small and medium, to keep track of the two smallest elements encountered so far.

    If the current element nums[i] is less than or equal to small, update small with nums[i].
    If nums[i] is greater than small but less than or equal to medium, update medium with nums[i].
    If nums[i] is greater than both small and medium, an increasing triplet was found (return true).
    Return False if the end of the array is reached without finding an increasing triplet.

    Caveat:
    *small* doesnt really mean the smalled element, but rather smaller than medium.

    Example:
    -------
        Consider the input [20,100,10,12,5,13],

        i = 0, smaller = inf, medium = inf
        i = 1, smaller 20, medium = inf
        i = 2, smaller 20, medium = 100
        i = 3, smaller 10, medium = 100
        i = 4, smaller 10, medium = 12
        i = 5, smaller 5, medium = 12
        observe that smaller is not at position after medium,
        which would breat the contraint we are interested in
        i < j < k and nums[i] < nums[j] < nums[k]

        However, the current smaller and the previous are still smaller than medium.
        And this is enough to guarantee the triplet exists if we find any element
        bigget than medium that is positioned after medium.

        At i = 6, num[i] = 13, which satisfies the triplet constraint
        i = 6, smaller 5, medium = 12

        Thus, return True

    """
    small, medium = float("inf"), float("inf")
    for e in nums:
        if e <= small:
            small = e
        elif e <= medium:
            medium = e
        else:
            return True
    return False
