"""
https://leetcode.com/problems/sum-of-subarray-minimums

Given an array of integers arr, find the sum of min(b),
where b ranges over every (contiguous) subarray of arr.
Since the answer may be large, return the answer modulo 109 + 7.

Example 1:

Input: arr = [3,1,2,4]
Output: 17
Explanation:
Subarrays are [3], [1], [2], [4], [3,1], [1,2], [2,4], [3,1,2], [1,2,4], [3,1,2,4].
Minimums are 3, 1, 2, 4, 1, 1, 2, 1, 1, 1.
Sum is 17.
Example 2:

Input: arr = [11,81,94,43,3]
Output: 444


Constraints:

1 <= arr.length <= 3 * 104
1 <= arr[i] <= 3 * 104
"""

from typing import List


class SumSubarrayMinsNaive:
    """O(n^2)"""

    def solve(self, arr: List[int]) -> int:
        n = len(arr)
        size = n - 1
        subtotal = min(arr)
        while size >= 1:
            start_p = 0
            offset = n - size
            while start_p + offset <= n:
                # print(arr[start_p: start_p + offset])
                subtotal = subtotal + min(arr[start_p : start_p + offset])
                start_p += 1
            size -= 1
        # preventing integer overflow and keeping the result within a manageable range:
        return subtotal % 1000000007  # (10 ** 9 + 7)


class SumSubarrayMinsStackingA:
    def solve(self, arr: List[int]) -> int:
        n = len(arr)
        # These arrays will hold for each element
        # the index of the previous smaller element (left) and
        # the next smaller or equal element (right).
        left_idx = [-1] * n
        right_idx = [n] * n

        # populate the left indexes array using a monotonic decreasing stack.
        # It iterates through each element in the array arr,
        # maintaining a stack that stores pairs of indices and values.
        # The stack is used to find the previous smaller element's index
        stack = []  # (idx, value)
        for idx, value in enumerate(arr):
            # If the current value is smaller than or equal to the top of the stack,
            # elements are popped from the stack until a smaller element is found or
            # the stack becomes empty. This guarantees a decreasing monotonic stack
            while stack and value <= stack[-1][1]:
                stack.pop()
            if stack:
                left_idx[idx] = stack[-1][0]
            stack.append((idx, value))  # (idx, value)

        # Collect the right indexes using the same approach
        # but in reverse order, ie, from right to left
        stack = []  # (idx, value)
        for idx in range(n - 1, -1, -1):
            while stack and arr[idx] < stack[-1][1]:
                stack.pop()
            if stack:
                right_idx[idx] = stack[-1][0]
            stack.append((idx, arr[idx]))  # (idx, value)

        min_value_frequency = [(i - left_idx[i]) * (right_idx[i] - i) * value for i, value in enumerate(arr)]
        subtotal = sum(min_value_frequency)
        return subtotal % 1000000007  # (10 ** 9 + 7)


class SumSubarrayMinsStackingB:
    """
    Explanation: https://youtu.be/aX1F2-DrBkQ?si=V3oBj-5eT6asMZ1w
    """

    def solve(self, arr: List[int]) -> int:
        subtotal = 0
        stack = []  # (index, value)
        # first phase: iterate over the input array
        for idx, value in enumerate(arr):
            while stack and value < stack[-1][1]:  # (idx, value)
                jdx, element = stack.pop()
                left_num_subarr = jdx - stack[-1][0] if stack else jdx + 1
                right_num_subarr = idx - jdx
                subtotal = subtotal + element * left_num_subarr * right_num_subarr
            stack.append((idx, value))

        # second phase: iterate over the stack
        for idx in range(len(stack)):
            jdx, element = stack[idx]
            left_num_subarr = jdx - stack[idx - 1][0] if idx > 0 else jdx + 1
            right_num_subarr = len(arr) - jdx
            subtotal = subtotal + element * left_num_subarr * right_num_subarr
        return subtotal % 1000000007  # (10 ** 9 + 7)
