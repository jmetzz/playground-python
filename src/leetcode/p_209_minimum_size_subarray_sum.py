"""
https://leetcode.com/problems/minimum-size-subarray-sum/description

209. Minimum Size Subarray Sum
Medium

Given an array of positive integers nums and a positive integer target,
return the minimal length of a subarray whose sum is greater than or
equal to target. If there is no such subarray, return 0 instead.



Example 1:
Input: target = 7, nums = [2,3,1,2,4,3]
Output: 2
Explanation: The subarray [4,3] has the minimal length under the problem constraint.

Example 2:
Input: target = 4, nums = [1,4,4]
Output: 1

Example 3:
Input: target = 11, nums = [1,1,1,1,1,1,1,1]
Output: 0


Constraints:
1 <= target <= 10^9
1 <= nums.length <= 10^5
1 <= nums[i] <= 10^4


Follow up: If you have figured out the O(n) solution,
try coding another solution of which the time complexity is O(n log(n)).
"""


def min_sub_array_len(target: int, nums: list[int]) -> int:
    answer = float("inf")
    curr_window_sum = 0
    left = 0
    for right in range(len(nums)):
        # expand the window:
        curr_window_sum += nums[right]
        while curr_window_sum >= target:
            # shrink the window:
            answer = min(answer, right - left + 1)
            curr_window_sum -= nums[left]
            left += 1
    return answer if answer != float("inf") else 0


if __name__ == "__main__":
    functions = [min_sub_array_len]
    inputs = [
        (5, [1, 1, 1], 0),
        (4, [1, 4, 4], 1),
        (7, [2, 3, 1, 2, 4, 3], 2),
        (15, [1, 2, 3, 4, 5], 5),
    ]
    for func in functions:
        for target, nums, expected in inputs:
            actual = func(target, nums)
            print(f"actual:{actual} ; expected: {expected} -> correct? {actual == expected}")
