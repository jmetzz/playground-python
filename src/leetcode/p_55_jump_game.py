"""https://leetcode.com/problems/jump-game/description

55. Jump Game
Medium
Companies
You are given an integer array nums. You are initially positioned at
the array's first index, and each element in the array represents
your maximum jump length at that position.

Return true if you can reach the last index, or false otherwise.



Example 1:

Input: nums = [2,3,1,1,4]
Output: true
Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.
Example 2:

Input: nums = [3,2,1,0,4]
Output: false
Explanation: You will always arrive at index 3 no matter what.
Its maximum jump length is 0, which makes it impossible to reach the last index.


Constraints:

1 <= nums.length <= 10^4
0 <= nums[i] <= 10^5
"""

from functools import lru_cache


def can_jump_recursion(nums: list[int]) -> bool:
    """Time Limit Exceeded"""
    n = len(nums)

    @lru_cache
    def _dfs(idx) -> bool:
        if idx == n - 1:
            return True
        if idx >= n or nums[idx] == 0:
            return False

        return any(_dfs(idx + off_set) for off_set in range(1, nums[idx] + 1))

    return _dfs(0)


def can_jump_loop(nums: list[int]) -> bool:
    """Use the distance traveling and gas fueling analogy"""
    gas_in_tank = 0
    for gas in nums:
        if gas_in_tank < 0:
            return False
        gas_in_tank = max(gas_in_tank, gas)

        # discount the fuel to move to this position
        gas_in_tank -= 1

    return True


def can_jump_greedy(nums: list[int]) -> bool:
    """Working backwards"""
    target_idx = len(nums) - 1
    for curr_idx in reversed(range(len(nums) - 1)):
        if curr_idx + nums[curr_idx] >= target_idx:
            target_idx = curr_idx
    return target_idx == 0


if __name__ == "__main__":
    functions = [can_jump_recursion, can_jump_loop, can_jump_greedy]
    for f in functions:
        print(f([2, 3, 1, 1, 4]))  # true
        print(f([3, 2, 1, 0, 4]))  # false
        print()
