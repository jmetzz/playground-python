"""
https://leetcode.com/problems/majority-element/description

169. Majority Element
Easy

Given an array nums of size n, return the majority element.

The majority element is the element that appears more than ⌊n / 2⌋ times.
You may assume that the majority element always exists in the array.

Example 1:
Input: nums = [3,2,3]
Output: 3

Example 2:
Input: nums = [2,2,1,1,1,2,2]
Output: 2

Constraints:
n == nums.length
1 <= n <= 5 * 104
-109 <= nums[i] <= 109


Follow-up: Could you solve the problem in linear time and in O(1) space?
"""
import heapq
import math
from collections import Counter, defaultdict
from typing import List, Optional


class MajorityElement:
    def solve_counter(self, nums: List[int]) -> Optional[int]:
        if not nums:
            return None
        n = len(nums)
        if n == 1:
            return nums[0]

        counter = defaultdict(int)
        most_frequent = nums[0]
        counter[most_frequent] = 1
        for i in range(1, n):
            e = nums[i]
            counter[e] += 1
            if counter[e] > math.floor(n / 2):
                return e
            if counter[most_frequent] > counter[e]:
                most_frequent = e

        strict = all([counter[most_frequent] > c for key, c in counter.items() if key != most_frequent])
        return most_frequent if strict else None

    def solve_heap(self, nums: List[int]) -> Optional[int]:
        if not nums:
            return None
        if len(nums) == 1:
            return nums[0]

        counter = Counter(nums)
        heap = [(-v, k) for k, v in counter.items()]
        heapq.heapify(heap)

        major = heapq.heappop(heap)
        sec = heapq.heappop(heap)

        return major[1] if major[0] < sec[0] else None

    def solve_boyer_moore_majority_alg(self, nums: List[int]) -> Optional[int]:
        # if not nums:
        #     return None

        major = None
        counter = 0
        for e in nums:
            if counter == 0:
                major = e
                counter = 1
            elif major == e:
                counter += 1
            else:
                counter -= 1

        """
        Even when the input sequence has no majority,
        the algorithm will report one of the sequence elements as its result.
        However, a second pass over the same input sequence will
        determine whether it is actually a majority.
        """
        counter = 0
        n = len(nums)
        for e in nums:
            if major == e:
                counter += 1
                # try early return if the counter has
                # already passed the floor(n / 2) condition
                if counter > math.floor(n / 2):
                    return major

        return None


if __name__ == "__main__":
    solver = MajorityElement()

    input_arr = [3, 2, 3]
    print(solver.solve_counter(input_arr))  # 3
    print(solver.solve_heap(input_arr))  # 3
    print(solver.solve_boyer_moore_majority_alg(input_arr))  # 3
