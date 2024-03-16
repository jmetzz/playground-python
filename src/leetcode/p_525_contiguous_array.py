"""
https://leetcode.com/problems/contiguous-array/description

525. Contiguous Array
Medium

Given a binary array nums, return the maximum length of a contiguous
subarray with an equal number of 0 and 1.



Example 1:
Input: nums = [0,1]
Output: 2
Explanation: [0, 1] is the longest contiguous subarray with an equal number of 0 and 1.

Example 2:
Input: nums = [0,1,0]
Output: 2
Explanation: [0, 1] (or [1, 0]) is a longest contiguous subarray with equal number of 0 and 1.

Constraints:
1 <= nums.length <= 105
nums[i] is either 0 or 1.
"""

from collections import defaultdict
from typing import List


def find_max_length_brute_force(nums: List[int]) -> int:
    """ "
    We consider every possible subarray and count the number
    of zeros and ones in each subarray. Then, find out the maximum
    size subarray with equal number of zeros and ones.

    Time: O(n^2)
    Space: O(1)
    """
    max_len = 0
    for start in range(0, len(nums)):
        zeros, ones = 0, 0
        for end in range(start, len(nums)):
            if nums[end] == 0:
                zeros += 1
            else:
                ones += 1
            if zeros == ones:
                max_len = max(max_len, end - start + 1)

    return max_len


def find_max_length_hashmap(nums: List[int]) -> int:
    """
    Use a proxy counter variable (count_diff) to keep the relative number of
    ones and zeros encountered so far while traversing the array.
    Increment count_diff when a 1 is found, and decrement it when a 0 is found.

    If count_diff becomes zero, if implies the number of 1s and 0s is equal
    in the subarray (from the beginning till the current index of the array).
    So, calculate the lenght of the subarray and collect the max so far.

    Note that if count_diff is repeated twice (or more) while traversing the array,
    it means that the number of zeros and ones are equal between the
    indices corresponding to the equal count_diff value occurences.
    Consider this sequence as example:
        [
            0
        A   0 -> diff -2
            1
        B   0 -> diff -2
            0
            0
            1
        C   1 -> diff -2
        ]

        lenght(A, B) = 2
        lenght(B, C) = 4
        lenght(A, C) = 6

    To simplify the solution, the idea is to track the first occurrence of each possible
    count_diff between the count of 1s and 0s observed as we traverse the array.

    For that, keep an auxiliry dictionary mapping count_diff to the index in the input array.

    Time: O(n)
    Space: O(n) --> Maximum size of the HashMap mapmapmap will be n, if all the elements are either 1 or 0

    """
    # track the first occurrence of a particular count of 1s and 0s.
    first_idx_diff_map = {}  # (diff_count -> index)
    first_idx_diff_map[0] = -1  # Base case: diff of 0 before starting

    max_length, count_diff = 0, 0
    for i in range(len(nums)):
        count_diff += 1 if nums[i] == 1 else -1
        # Calculate the adjusted index in first_diff_index for the current count_diff
        if count_diff not in first_idx_diff_map:
            # this difference has NOT been observed before,
            # i is the first occurrence of this difference.
            first_idx_diff_map[count_diff] = i
        else:
            # this difference has been observed before,
            # and a subarray that balances 1s and 0s
            # might end at the current index i
            max_length = max(max_length, i - first_idx_diff_map[count_diff])

    return max_length


def findMaxLength(nums: List[int]) -> int:
    ans = curr = 0
    counts = defaultdict(int)
    counts[0] = -1

    for i in range(len(nums)):
        curr += 1 if nums[i] == 1 else -1
        if curr in counts:
            ans = max(ans, i - counts[curr])
        else:
            counts[curr] = i

    return ans


if __name__ == "__main__":
    print(find_max_length_hashmap([0, 1]))
    print(find_max_length_hashmap([0, 1, 0]))
    print(find_max_length_hashmap([0, 1, 0, 0, 1, 1, 0]))
