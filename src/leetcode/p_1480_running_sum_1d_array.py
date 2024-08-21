"""https://leetcode.com/problems/running-sum-of-1d-array/description/

1480. Running Sum of 1d Array
Easy

Given an array nums. We define a running sum of an array as runningSum[i] = sum(nums[0]…nums[i]).

Return the running sum of nums.

Example 1:
Input: nums = [1,2,3,4]
Output: [1,3,6,10]
Explanation: Running sum is obtained as follows: [1, 1+2, 1+2+3, 1+2+3+4].

Example 2:
Input: nums = [1,1,1,1,1]
Output: [1,2,3,4,5]
Explanation: Running sum is obtained as follows: [1, 1+1, 1+1+1, 1+1+1+1, 1+1+1+1+1].

Example 3:
Input: nums = [3,1,2,10,1]
Output: [3,4,6,16,17]


Constraints:
1 <= nums.length <= 1000
-10^6 <= nums[i] <= 10^6
"""

from typing import List


def running_sum_1d(nums: List[int]) -> List[int]:
    answer = [0] * len(nums)
    curr_sum = 0
    for i, value in enumerate(nums):
        curr_sum += value
        answer[i] = curr_sum
    return answer


def running_sum_1d_2(nums: List[int]) -> List[int]:
    answer = [0] * len(nums)
    answer[0] = nums[0]
    for i in range(1, len(nums)):
        answer[i] = nums[i] + answer[i - 1]
    return answer


if __name__ == "__main__":
    print(running_sum_1d([1, 3, 6, 10]))
    print(running_sum_1d_2([1, 3, 6, 10]))
