"""
https://leetcode.com/problems/find-all-duplicates-in-an-array/description

442. Find All Duplicates in an Array
Medium

Given an integer array nums of length n where all the integers of nums are in the range [1, n] and
each integer appears once or twice, return an array of all the integers that appears twice.

You must write an algorithm that runs in O(n) time and uses only constant extra space.


Example 1:
Input: nums = [4,3,2,7,8,2,3,1]
Output: [2,3]

Example 2:
Input: nums = [1,1,2]
Output: [1]

Example 3:
Input: nums = [1]
Output: []

Constraints:
n == nums.length
1 <= n <= 105
1 <= nums[i] <= n
Each element in nums appears once or twice.
"""

from typing import Counter, List


def find_duplicates(nums: List[int]) -> List[int]:
    counter = Counter(nums)
    return [e for e, qtt in counter.items() if qtt > 1]


def find_duplicates_space_optimized(nums: List[int]) -> List[int]:
    """

    To solve this problem in O(n) time and constant extra space,
    take advantage of the constraints:
      * all integers in the array are in the range [1, n]
      * each integer appears once or twice.

    Iterate through the array and use each element as an index to mark the corresponding element as seen.
    This is done by making it nevatige (* -1).
    Then, as the iteration over the array progresses, anytime you find a negative element
    you are sur this element was already seen.

    Time complexity: O(n)
    Space complexity: O(k), where k is the number of duplicate elements maxing at k = n/2
    """
    answer = set()
    for e in nums:
        e = abs(e)
        if nums[e - 1] < 0:
            answer.add(e)
        else:
            nums[e - 1] *= -1
    return list(answer)
