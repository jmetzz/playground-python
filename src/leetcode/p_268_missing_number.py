"""

https://leetcode.com/problems/missing-number/

268. Missing Number
Solved
Easy
Topics
Companies
Given an array nums containing n distinct numbers in the range [0, n],
return the only number in the range that is missing from the array.



Example 1:
Input: nums = [3,0,1]
Output: 2
Explanation: n = 3 since there are 3 numbers, so all numbers are in the range [0,3].
2 is the missing number in the range since it does not appear in nums.

Example 2:
Input: nums = [0,1]
Output: 2
Explanation: n = 2 since there are 2 numbers, so all numbers are in the range [0,2].
2 is the missing number in the range since it does not appear in nums.

Example 3:
Input: nums = [9,6,4,2,3,5,7,0,1]
Output: 8
Explanation: n = 9 since there are 9 numbers, so all numbers are in the range [0,9].
8 is the missing number in the range since it does not appear in nums.


Constraints:

n == nums.length
1 <= n <= 104
0 <= nums[i] <= n
All the numbers of nums are unique.


Follow up: Could you implement a solution using only O(1) extra space complexity and O(n) runtime complexity?
"""
from typing import List


def missing_number_solve_set(nums: List[int]) -> int:
    n = len(nums)
    diff = set(range(n + 1)) - set(nums)
    return diff.pop()


def missing_number_solve_naive_loops(nums: List[int]) -> int:
    n = len(nums)
    v = [-1] * (n + 1)
    for num in nums:
        v[num] = num
    for i in range(len(v)):
        if v[i] == -1:
            return i
    return 0


def missing_number_solve_xor(nums: List[int]) -> int:
    """

    XOR operations reminder:
        0 XOR 5 = 5
        5 XOR 0 = 5
        5 XOR 5 = 0
        4 XOR 5 = 4 XOR 5

    input size n: 3
    XOR mask: [0, 1, 2, 3]
        -->    0^1: 1
        -->         1^2: 3
        -->              3^3: 0
        ----------------------^
    input:   [3, 0, 1]
        -->   3^0: 3
        -->        3^0: 3
        -->             3^1: 2
    ans: 2 ------------------^
    """
    n = len(nums)
    ans = 0
    for i in range(1, n + 1):
        ans ^= i
    for num in nums:
        ans ^= num
    return ans


def missing_number_solve_sum(nums: List[int]) -> int:
    """
    1. sum all the elements in the interval [0, n]
        sum[0, n] = n * (n + 1) / 2
    2. sum all elements in nums
    3. take the difference
    """
    n = len(nums)
    expected_total = int((n * (n + 1)) / 2)
    return expected_total - sum(nums)


def missing_number_solve_sorting(nums: List[int]) -> int:
    nums.sort()
    n = len(nums)
    if nums[0] != 0:
        return 0

    if nums[-1] != n:
        return n

    for i in range(1, len(nums)):
        if nums[i] != i:
            return i

    return 0
