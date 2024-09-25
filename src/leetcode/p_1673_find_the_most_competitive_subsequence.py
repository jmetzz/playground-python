"""https://leetcode.com/problems/find-the-most-competitive-subsequence/description/

1673. Find the Most Competitive Subsequence
Medium

Given an integer array nums and a positive integer k, return the most competitive subsequence of nums of size k.

An array's subsequence is a resulting sequence obtained by erasing some (possibly zero) elements from the array.

We define that a subsequence a is more competitive than a subsequence b (of the same length) if in the
first position where a and b differ, subsequence a has a number less than the corresponding number in b.
For example, [1,3,4] is more competitive than [1,3,5] because the first position they differ is at
the final number, and 4 is less than 5.


Example 1:
Input: nums = [3,5,2,6], k = 2
Output: [2,6]
Explanation: Among the set of every possible subsequence: {[3,5], [3,2], [3,6], [5,2], [5,6], [2,6]},
[2,6] is the most competitive.

Example 2:
Input: nums = [2,4,3,3,5,4,9,6], k = 4
Output: [2,3,3,4]


Constraints:

1 <= nums.length <= 105
0 <= nums[i] <= 109
1 <= k <= nums.length
"""


def most_competitive_1(nums: list[int], k: int) -> list[int]:
    """Given an array nums and a size k, find a subsequence of length k that is the lexicographically smallest.

    A subsequence is a sequence that can be derived from the array by deleting
    some or no elements without changing the order of the remaining elements.

    The intuitive approach involves the use of a monotonic stack to keep track
    of the smallest sequence formed so far, by pushing elements onto the stack
    if they help in forming the sequence.
    We also need to keep track of the size of the stack, since we need exactly k elements.
    If pushing a new element would exceed k, and there are still elements left in nums
    to consider, you might need to pop the last element if it's larger than the
    current element being considered.

    We can also use an auxiliary variable to count the number of deletions,
    and use it to control how many elements are still available in nums that
    could be moved directly onto the stack.
    This approche will make the stack bigger than what we want, though, which might
    be wasteful in terms of space.

    Alternativelly, we could only add elements onto the stack if there is still space.
    However, this extra condition (and calculating it for every iteration) will make the
    code slower.
    """
    stack = []
    delete_attempts = len(nums) - k
    for e in nums:
        while stack and stack[-1] > e and delete_attempts > 0:
            stack.pop()
            delete_attempts -= 1
        stack.append(e)
    return stack[:k]


def most_competitive_2(nums, k):
    stack = []
    n = len(nums)
    for idx, e in enumerate(nums):
        while stack and e < stack[-1] and len(stack) + n - idx - 1 >= k:
            stack.pop()
        if len(stack) < k:
            stack.append(e)
    return stack
