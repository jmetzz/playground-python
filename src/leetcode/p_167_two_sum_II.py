"""https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/description/
167. Two Sum II - Input Array Is Sorted
Medium

Given a 1-indexed array of integers numbers that is already sorted in non-decreasing order,
find two numbers such that they add up to a specific target number.
Let these two numbers be
numbers[index1] and numbers[index2]
where 1 <= index1 < index2 <= numbers.length.

Return the indices of the two numbers, index1 and index2,
added by one as an integer array [index1, index2] of length 2.

The tests are generated such that there is exactly one solution.
You may not use the same element twice.

Your solution must use only constant extra space.


Example 1:
Input: numbers = [2,7,11,15], target = 9
Output: [1,2]
Explanation: The sum of 2 and 7 is 9. Therefore, index1 = 1, index2 = 2. We return [1, 2].

Example 2:
Input: numbers = [2,3,4], target = 6
Output: [1,3]
Explanation: The sum of 2 and 4 is 6. Therefore index1 = 1, index2 = 3. We return [1, 3].

Example 3:
Input: numbers = [-1,0], target = -1
Output: [1,2]
Explanation: The sum of -1 and 0 is -1. Therefore index1 = 1, index2 = 2. We return [1, 2].


Constraints:

2 <= numbers.length <= 3 * 10^4
-1000 <= numbers[i] <= 1000
numbers is sorted in non-decreasing order.
-1000 <= target <= 1000
The tests are generated such that there is exactly one solution.
"""

from bisect import bisect


def two_sum_brute_force(numbers: list[int], target: int) -> list[int]:
    """The tests are generated such that there is exactly one solution.
    Time complexity O(n^2)
    Space Complexity O(1)
    """
    n = len(numbers)
    for i in range(n):  # noqa: RET503
        for j in range(i + 1, n):
            if numbers[i] + numbers[j] == target:
                return [i + 1, j + 1]


def two_sum_bisect(numbers: list[int], target: int) -> list[int]:
    """The tests are generated such that there is exactly one solution.
    Time Complexity O(n log n)
    Space Complexity O(1)
    """
    if len(numbers) == 2:
        return [1, 2]

    for i, e in enumerate(numbers):  # noqa: RET503
        complement = target - e
        candidate_idx = bisect(numbers, complement)
        if candidate_idx <= len(numbers) and complement == numbers[candidate_idx - 1]:
            return [i + 1, candidate_idx]


def two_sum_twopointers(numbers: list[int], target: int) -> list[int]:
    """Use two pointer to check the array only once
    Time Complexity O(n)
    Space Complexity O(1)
    """
    n = len(numbers)
    left, right = 0, n - 1

    while left < right:
        _sum = numbers[left] + numbers[right]
        if _sum == target:
            return [left + 1, right + 1]
        if _sum > target:
            # decrease the candidate sum
            right -= 1
        else:
            # increase the candidate sum
            left += 1
    return [-1, -1]  # invalid solution


if __name__ == "__main__":
    functions = [two_sum_brute_force, two_sum_bisect, two_sum_twopointers]

    for func in functions:
        print(func(numbers=[-1, 0], target=-1))  # Output: [1,2]
        print(func(numbers=[2, 3, 4], target=6))  # Output: [1,3]
        print(func(numbers=[2, 7, 11, 15], target=9))  # Output: [1,2]
        print(func(numbers=[1, 3, 4, 5, 7, 10, 11], target=9))  # Output: [3,4]
        print()
