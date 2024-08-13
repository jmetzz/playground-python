"""
https://leetcode.com/problems/rotate-array/description

189. Rotate Array
Medium

Given an integer array nums, rotate the array to the right by k steps, where k is non-negative.

Example 1:
Input: nums = [1,2,3,4,5,6,7], k = 3
Output: [5,6,7,1,2,3,4]
Explanation:
rotate 1 steps to the right: [7,1,2,3,4,5,6]
rotate 2 steps to the right: [6,7,1,2,3,4,5]
rotate 3 steps to the right: [5,6,7,1,2,3,4]

Example 2:
Input: nums = [-1,-100,3,99], k = 2
Output: [3,99,-1,-100]
Explanation:
rotate 1 steps to the right: [99,-1,-100,3]
rotate 2 steps to the right: [3,99,-1,-100]


Constraints:
1 <= nums.length <= 105
-231 <= nums[i] <= 231 - 1
0 <= k <= 105


Follow up:
Try to come up with as many solutions as you can. There are at least three different ways to solve this problem.
Could you do it in-place with O(1) extra space?
"""


def rotate_inplace_1(nums: list[int], k: int) -> None:
    """
    Do not return anything, modify nums in-place instead.

    This implementation will hit Time Limit Exceeded.
    """
    for _ in range(k):
        temp = nums[-1]
        for i in reversed(range(len(nums))):
            nums[i] = nums[i - 1]
        nums[0] = temp


def rotate_inplace_2(nums: list[int], k: int) -> None:
    """
    Do not return anything, modify nums in-place instead.

    Approach:
    1. calculate the effective rotation steps k by taking the modulus of k with
       the size of the array n. This ensures that k is within the range [0, n),
       which is necessary because rotating by a multiple of the list length is
       equivalent to not rotating at all.
    2. reverse the entire array to effectively moves the original last k elements to the front.
    3. Reverse only the first k elements to bring them to their correct positions.
    4 Reverse the rest of the elements in the array.
    """

    k = k % len(nums)  # Ensure k is within the range [0, n)

    nums.reverse()
    # rewrite the reversed portions
    nums[:k] = nums[:k][::-1]
    nums[k:] = nums[k:][::-1]


if __name__ == "__main__":
    nums = [1, 2, 3, 4, 5, 6, 7]
    rotate_inplace_1(nums, 3)
    print(nums)  # [5,6,7,1,2,3,4]

    nums = [-1, -100, 3, 99]
    rotate_inplace_1(nums, 2)
    print(nums)  # [3,99,-1,-100]

    nums = [1, 2, 3, 4, 5, 6, 7]
    rotate_inplace_2(nums, 3)
    print(nums)  # [5,6,7,1,2,3,4]

    nums = [-1, -100, 3, 99]
    rotate_inplace_2(nums, 2)
    print(nums)  # [3,99,-1,-100]
