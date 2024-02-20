"""
https://leetcode.com/problems/divide-array-into-arrays-with-max-difference/description

2966. Divide Array Into Arrays With Max Difference
Medium

You are given an integer array nums of size n and a positive integer k.

Divide the array into one or more arrays of size 3 satisfying the following conditions:

Each element of nums should be in exactly one array.
The difference between any two elements in one array is less than or equal to k.
Return a 2D array containing all the arrays. If it is impossible to satisfy the conditions,
return an empty array. And if there are multiple answers, return any of them.



Example 1:
Input: nums = [1,3,4,8,7,9,3,5,1], k = 2
Output: [[1,1,3],[3,4,5],[7,8,9]]
Explanation: We can divide the array into the following arrays: [1,1,3], [3,4,5] and [7,8,9].
The difference between any two elements in each array is less than or equal to 2.
Note that the order of elements is not important.

Example 2:
Input: nums = [1,3,3,2,7,3], k = 3
Output: []
Explanation: It is not possible to divide the array satisfying all the conditions.


Constraints:

n == nums.length
1 <= n <= 10^5
n is a multiple of 3.
1 <= nums[i] <= 10^5
1 <= k <= 10^5
"""
from typing import List


class DivideArraysWithMaxDiff:
    def solve_backward(self, nums: List[int], k: int) -> List[List[int]]:
        size = len(nums)
        if size % 3 != 0:
            return []

        sorted_nums = sorted(nums)
        answer = [None] * (size // 3)

        bucket_index = len(answer) - 1
        i = size - 1
        while i >= 2:
            # check diff
            if (
                sorted_nums[i] - sorted_nums[i - 1] > k
                or sorted_nums[i] - sorted_nums[i - 2] > k
                or sorted_nums[i - 2] - sorted_nums[i - 1] > k
            ):
                return []

            answer[bucket_index] = [
                sorted_nums[i],
                sorted_nums[i - 1],
                sorted_nums[i - 2],
            ]
            bucket_index -= 1
            i -= 3
        return answer

    def solve_forward(self, nums: List[int], k: int) -> List[List[int]]:
        size = len(nums)
        if size % 3 != 0:
            return []

        sorted_nums = sorted(nums)

        answer = []
        for i in range(0, size, 3):
            if i + 2 >= size or sorted_nums[i + 2] - sorted_nums[i] > k:
                return []
            answer.append([sorted_nums[i], sorted_nums[i + 1], sorted_nums[i + 2]])

        return answer
