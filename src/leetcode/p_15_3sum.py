"""
https://leetcode.com/problems/3sum

15. 3Sum
Medium

Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]]
such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

Notice that the solution set must not contain duplicate triplets.

Example 1:
Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]
Explanation:
nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0.
nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0.
nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0.
The distinct triplets are [-1,0,1] and [-1,-1,2].
Notice that the order of the output and the order of the triplets does not matter.

Example 2:
Input: nums = [0,1,1]
Output: []
Explanation: The only possible triplet does not sum up to 0.

Example 3:
Input: nums = [0,0,0]
Output: [[0,0,0]]
Explanation: The only possible triplet sums up to 0.


Constraints:
3 <= nums.length <= 3000
-10^5 <= nums[i] <= 10^5
"""


def three_sum_sorting(nums: list[int]) -> list[list[int]]:
    nums.sort()
    answer = set()
    n = len(nums)
    for idx in range(n - 2):
        left, right = idx + 1, n - 1
        target = nums[idx]
        while left < right:
            _sum = nums[left] + nums[right] + target
            if _sum == 0:
                answer.add((nums[idx], nums[left], nums[right]))
                left += 1
                right -= 1
            elif _sum > 0:
                right -= 1
            else:
                left += 1
    return answer


def three_sum_speedup(nums: list[int]) -> list[list[int]]:
    nums.sort()
    answer = []
    n = len(nums)
    for idx in range(n - 2):
        if idx > 0 and nums[idx] == nums[idx - 1]:
            # skip duplicate elements in the sorted array.
            # ensures that the solution does not consider
            # the same triplet multiple times
            continue

        left, right = idx + 1, n - 1
        target = nums[idx]
        while left < right:
            _sum = nums[left] + nums[right] + target
            if _sum == 0:
                answer.append([nums[idx], nums[left], nums[right]])
                left += 1
                right -= 1

                # move the two pointers to skip unnecessary calculation
                while left < right and nums[left] == nums[left - 1]:
                    left += 1

                while left < right and nums[right] == nums[right + 1]:
                    right -= 1

            elif _sum > 0:
                right -= 1
            else:
                left += 1

    return answer


if __name__ == "__main__":
    functions = [three_sum_sorting, three_sum_speedup]
    for func in functions:
        print(func([0, 0, 0]))  # Output: [[0,0,0]]
        print(func([0, 0, 0, 0]))  # Output: [[0,0,0]]
        print(func([0, 1, 1]))  # Output: []
        print(func([-1, 0, 1, 2, -1, -4]))  # Output: [[-1,-1,2],[-1,0,1]]
        print(func([-2, 0, 1, 1, 2]))  # Output: [[-2,0,2],[-2,1,1]]
        print(
            func([-1, 0, 1, 2, -1, -4, -2, -3, 3, 0, 4])
        )  # Output: [[-4,0,4],[-4,1,3],[-3,-1,4],[-3,0,3],[-3,1,2],[-2,-1,3],[-2,0,2],[-1,-1,2],[-1,0,1]]
        print()
