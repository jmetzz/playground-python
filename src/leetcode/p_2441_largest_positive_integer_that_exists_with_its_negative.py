"""https://leetcode.com/problems/largest-positive-integer-that-exists-with-its-negative/description
2441. Largest Positive Integer That Exists With Its Negative

Given an integer array nums that does not contain any zeros,
find the largest positive integer k such that -k also exists in the array.

Return the positive integer k. If there is no such integer, return -1.



Example 1:
Input: nums = [-1,2,-3,3]
Output: 3
Explanation: 3 is the only valid k we can find in the array.

Example 2:
Input: nums = [-1,10,6,7,-7,1]
Output: 7
Explanation: Both 1 and 7 have their corresponding negative values in the array.
7 has a larger value.

Example 3:
Input: nums = [-10,8,6,7,-2,-3]
Output: -1
Explanation: There is no a single valid k, we return -1.


Constraints:

1 <= nums.length <= 1000
-1000 <= nums[i] <= 1000
nums[i] != 0

"""


def find_max_k__set_intersection(nums: list[int]) -> int:
    negatives = set([-e for e in nums if e < 0])
    intersec = set(nums) & negatives
    if intersec:
        return sorted(intersec, reverse=True)[0]
    return -1


def find_max_k__set_removal(nums: list[int]) -> int:
    elements = set(nums)
    while elements:
        max_value = max(elements)
        if -max_value in elements:
            return max_value
        elements.remove(max_value)
    return -1


if __name__ == "__main__":
    print(find_max_k__set_removal([-1, 2, -3, 3]))  # 3
    print(find_max_k__set_removal([-1, 10, 6, 7, -7, 1]))  # 7
    print(find_max_k__set_removal([-10, 8, 6, 7, -2, -3]))  # - 1
