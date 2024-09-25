"""https://leetcode.com/problems/max-number-of-k-sum-pairs/description/

1679. Max Number of K-Sum Pairs
Medium

You are given an integer array nums and an integer k.

In one operation, you can pick two numbers from the array whose sum equals k and remove them from the array.

Return the maximum number of operations you can perform on the array.


Example 1:

Input: nums = [1,2,3,4], k = 5
Output: 2
Explanation: Starting with nums = [1,2,3,4]:
- Remove numbers 1 and 4, then nums = [2,3]
- Remove numbers 2 and 3, then nums = []
There are no more pairs that sum up to 5, hence a total of 2 operations.

Example 2:
Input: nums = [3,1,3,4,3], k = 6
Output: 1
Explanation: Starting with nums = [3,1,3,4,3]:
- Remove the first two 3's, then nums = [1,4,3]
There are no more pairs that sum up to 6, hence a total of 1 operation.


Constraints:

1 <= nums.length <= 105
1 <= nums[i] <= 109
1 <= k <= 109
"""

from collections import Counter


def max_operations_two_pointers(nums: list[int], k: int) -> int:
    """Iterate with two pointers: Use a while loop to iterate until left pointer is less than right pointer.

    - If the sum of nums[left] and nums[right] is equal to the target k, we have found a pair,
    and thus increment the operation count and move both pointers towards the center
    - If the sum is less than k, we need a larger sum, so we advance the left pointer.
    - If the sum is greater than k, we need a smaller sum, therefore, we move the right pointer to the left.

    After the loop completes, return the operation count.

    Complexity
    Time complexity: O(n log n) due to the sorting step

    Space complexity: O(n) due the the copy of the input array during sorting (This solution is not inplace).
    """
    if not nums:
        return 0

    sorted_nums = sorted(nums)
    op_counter = 0
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        if sorted_nums[lo] + sorted_nums[hi] == k:
            op_counter += 1
            lo += 1
            hi -= 1
        elif sorted_nums[lo] + sorted_nums[hi] < k:
            lo += 1
        else:
            hi -= 1
    return op_counter


def max_operations_counter(nums: list[int], k: int) -> int:
    if not nums:
        return 0

    freq_map = Counter(nums)
    operations = 0
    for num in freq_map:
        complement = k - num
        if complement == num:
            # If the number is its own complement, we can only form a pair if there's at least 2.
            # We don't need to explicitly decrement freq_map[num] in this case.
            # Since the loop guarantees num will not be revisited (keys in the dictionary are unique)
            operations += freq_map[num] // 2
        elif complement in freq_map:
            # The number of pairs is limited by the lesser occurrence of the num or its complement
            possible_ops = min(freq_map[num], freq_map[complement])
            operations += possible_ops
            # Decrement the frequencies since those pairs are now used
            freq_map[num] -= possible_ops
            freq_map[complement] -= possible_ops
    return operations
