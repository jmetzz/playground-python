"""
https://leetcode.com/problems/squares-of-a-sorted-array/description

977. Squares of a Sorted Array
Easy

Given an integer array nums sorted in non-decreasing order, return an array of the squares
of each number sorted in non-decreasing order.


Example 1:
Input: nums = [-4,-1,0,3,10]
Output: [0,1,9,16,100]
Explanation: After squaring, the array becomes [16,1,0,9,100].
After sorting, it becomes [0,1,9,16,100].

Example 2:
Input: nums = [-7,-3,2,3,11]
Output: [4,9,9,49,121]


Constraints:
1 <= nums.length <= 104
-104 <= nums[i] <= 104
nums is sorted in non-decreasing order.

Follow up: Squaring each element and sorting the new array is very trivial,
could you find an O(n) solution using a different approach?
"""

from bisect import bisect
from typing import List


def sorted_squares_naive(nums: List[int]) -> List[int]:
    return sorted([e * e for e in nums])


def sorted_squares_expanding_window(nums: List[int]) -> List[int]:
    n = len(nums)
    answer = []
    # all(elem <= x for elem in a[lo : ip]) is true for the left slice and
    # all(elem > x for elem in a[ip : hi]) is true for the right slice.
    right = bisect(nums, 0)
    left = right - 1

    while right < n and left >= 0:
        right_sqr = nums[right] * nums[right]
        left_sqr = nums[left] * nums[left]
        if right_sqr < left_sqr:
            answer.append(right_sqr)
            right += 1
        else:
            answer.append(left_sqr)
            left -= 1
    if left < 0:
        answer += [e * e for e in nums[right:]]
    elif right >= n:
        answer += [nums[i] * nums[i] for i in range(left, -1, -1)]
    return answer


def sorted_squares_shrinking_window(nums: List[int]) -> List[int]:
    answer = [0] * len(nums)
    left, right = 0, len(nums) - 1
    for i in range(len(nums) - 1, -1, -1):
        if abs(nums[left]) <= abs(nums[right]):
            square = nums[right] * nums[right]
            right -= 1
        else:
            square = nums[left] * nums[left]
            left += 1
        answer[i] = square
    return answer


if __name__ == "__main__":
    inputs = [
        [-5, -3, -2, -1],
        [-5, -3, -2, -1, 0],
        [-6, -5, -3, -2, -1],
        [-4, -1, 0, 3, 10],
        [-7, -3, 2, 3, 11],
    ]

    for arr in inputs:
        print(sorted_squares_naive(arr))
        print(sorted_squares_expanding_window(arr))
        print(sorted_squares_shrinking_window(arr))
        print()
