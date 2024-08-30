"""
https://leetcode.com/problems/longest-consecutive-sequence/description

128. Longest Consecutive Sequence
Medium

Given an unsorted array of integers nums,
return the length of the longest consecutive elements sequence.

You must write an algorithm that runs in O(n) time.

Example 1:
Input: nums = [100,4,200,1,3,2]
Output: 4
Explanation: The longest consecutive elements sequence is [1, 2, 3, 4].
Therefore its length is 4.

Example 2:
Input: nums = [0,3,7,2,5,8,4,6,0,1]
Output: 9

Constraints:
0 <= nums.length <= 10^5
-10^9 <= nums[i] <= 10^9
"""

from data_structures.graphs import UnionFind


def longest_consecutive_naive(nums: list[int]) -> int:
    """
    breaking the rule of the problem description: using sorting

    Time Complexity O(n logn)
    """
    if not nums:
        return 0
    # Use set to remove duplicates before sorting
    sorted_nums = sorted(set(nums))

    n = len(sorted_nums)
    max_streak = 1
    curr_streak = 1

    for i in range(1, n):
        if sorted_nums[i] == sorted_nums[i - 1] + 1:
            curr_streak += 1
        else:
            max_streak = max(max_streak, curr_streak)
            curr_streak = 1
    return max(max_streak, curr_streak)


def longest_consecutive(nums: list[int]) -> int:
    nums_set = set(nums)
    max_streak = 0
    for e in nums:
        if e - 1 not in nums_set:
            # e is the start of a sequence
            curr_streak = 1  # accounts for e
            while e + curr_streak in nums_set:
                curr_streak += 1
            max_streak = max(max_streak, curr_streak)
    return max_streak


def longest_consecutive_perf_improvement(nums: list[int]) -> int:
    nums_set = set(nums)
    longest_streak = 0
    for e in nums:
        if e - 1 not in nums_set:
            next_e = e + 1
            while next_e in nums_set:
                next_e += 1
            curr_streak = next_e - e
            longest_streak = max(longest_streak, curr_streak)
    return longest_streak


def longest_consecutive_union_find(nums: list[int]) -> int:
    if not nums:
        return 0

    # Map each number to its index
    num_to_index = {num: i for i, num in enumerate(nums)}
    uf = UnionFind(len(nums))

    # Union consecutive numbers
    for num in nums:
        if num + 1 in num_to_index:
            uf.unite(num_to_index[num], num_to_index[num + 1])

    # Find the size of the largest connected component
    root_to_size = {}
    for i in range(len(nums)):
        root = uf.find(i)
        if root not in root_to_size:
            root_to_size[root] = 0
        root_to_size[root] += 1

    return max(root_to_size.values())


if __name__ == "__main__":
    arr = [0, 3, 7, 2, 5, 8, 4, 6, 0, 1]
    print(longest_consecutive_naive([]))
    print(longest_consecutive([]))
    print(longest_consecutive_perf_improvement([]))
    print(longest_consecutive_union_find([]))

    print(longest_consecutive_naive(arr))
    print(longest_consecutive(arr))
    print(longest_consecutive_perf_improvement(arr))
    print(longest_consecutive_union_find(arr))
