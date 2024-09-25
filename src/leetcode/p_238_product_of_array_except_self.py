"""https://leetcode.com/problems/product-of-array-except-self/description/

238. Product of Array Except Self
Medium

Given an integer array nums, return an array answer such that answer[i]
is equal to the product of all the elements of nums except nums[i].

The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.

You must write an algorithm that runs in O(n) time and without using the division operation.



Example 1:
Input: nums = [1,2,3,4]
Output: [24,12,8,6]

Example 2:
Input: nums = [-1,1,0,-3,3]
Output: [0,0,9,0,0]


Constraints:

2 <= nums.length <= 105
-30 <= nums[i] <= 30
The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.


Follow up: Can you solve the problem in O(1) extra space complexity?
(The output array does not count as extra space for space complexity analysis.)
"""


def product_except_self(nums: list[int]) -> list[int]:
    n = len(nums)
    answer = [1] * n

    # calculate the prefix multiplication for each element i
    # and store in answer[i + 1]
    prefix = 1
    for i in range(n):
        # forward pass
        answer[i] = prefix
        prefix *= nums[i]

    # as we calculate the suffix mutiplication,
    # we can update the answer array as follows:
    # answer[i] = answer[i-1] * suffix
    suffix = 1
    for i in range(n - 1, -1, -1):
        # backward pass
        answer[i] *= suffix
        suffix *= nums[i]

    return answer
