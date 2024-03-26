"""
https://leetcode.com/problems/set-mismatch/

You have a set of integers s, which originally contains all the numbers from 1 to n. Unfortunately,
due to some error, one of the numbers in s got duplicated to another number in the set,
which results in repetition of one number and loss of another number.

You are given an integer array nums representing the data status of this set after the error.

Find the number that occurs twice and the number that is missing and return them in the form of an array.

Example 1:

Input: nums = [1,2,2,4]
Output: [2,3]
Example 2:

Input: nums = [1,1]
Output: [1,2]


Constraints:

2 <= nums.length <= 104
1 <= nums[i] <= 104
"""

from collections import Counter
from typing import List


class SetMismatch:
    def solve_with_setandsum(self, nums: List[int]) -> List[int]:
        # find the double element
        temp_set = set()
        double = -1
        for v in nums:
            if v in temp_set:
                double = v
                break
            temp_set.add(v)

        # use expected sum to find the missing element
        # this assumes no duplicates and no missing numbers.
        n = len(nums)
        expected_sum = int((n * (n + 1)) / 2)
        missing = expected_sum - (sum(nums) - double)

        return [double, missing]

    def solve_with_setandsum_2(self, nums: List[int]) -> List[int]:
        n = len(nums)
        # this assumes no duplicates and no missing numbers.
        expected_sum = int((n * (n + 1)) / 2)
        unique_sum = sum(set(nums))
        actual_sum = sum(nums)

        missing = expected_sum - unique_sum
        double = actual_sum - unique_sum
        return [double, missing]

    def solve_with_maps(self, nums: List[int]) -> List[int]:
        # map mp to count the occurrences of numbers from 1 to n, which is the domain of this problem
        frequency = Counter(range(1, len(nums) + 1))

        # iterate over the input numbers and mark the elements
        # visited, decreasing 1 from its frequency
        for value in nums:
            frequency[value] -= 1

        # find the missing and duplicate elements
        # Remember, the element with frequency -1 is missing,
        # and the element with frequency 1 is the duplicate
        double, missing = 0, 0
        for key, value in frequency.items():
            if value == -1:
                double = key
            if value == 1:
                missing = key

        return [double, missing]

    def solve_with_xor_ops(self, nums: List[int]) -> List[int]:
        # Calculate the XOR of all numbers from 1 to n (denoted as xorAll)
        # and the XOR of the array nums (denoted as xorArray).
        # Then, XOR the xorAll and xorArray to obtain xorResult.
        n = len(nums)
        xor_all = 0
        xor_arr = 0
        for i in range(1, n + 1):
            xor_all ^= i

        for num in nums:
            xor_arr ^= num

        xor_result = xor_arr ^ xor_all

        # Find the rightmost set bit in xorResult and store it in rightmostSetBit.
        rightmost_setbit = xor_result & -xor_result

        # Divide the numbers from 1 to n into two groups based on the rightmost set bit:
        # xorSet for numbers with the set bit, and xorNotSet for numbers without the set bit.
        xor_set = 0
        xor_not_set = 0

        for i in range(1, n + 1):
            if (i & rightmost_setbit) != 0:
                xor_set ^= i
            else:
                xor_not_set ^= i

        # XOR all numbers in nums with the rightmost set bit to find the duplicate and missing numbers.
        for num in nums:
            if (num & rightmost_setbit) != 0:
                xor_set ^= num
            else:
                xor_not_set ^= num

        # Iterate through nums and compare each number with xorSet.
        for num in nums:
            if num == xor_set:
                return [xor_set, xor_not_set]

        # If found, return {xorSet, xorNotSet}; otherwise, return {xorNotSet, xorSet}.
        return [xor_not_set, xor_set]
