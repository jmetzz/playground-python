"""

"""
from collections import defaultdict, deque
from typing import List


class DailyTemperatures:
    def solve_iterative(self, temperatures: List[int]) -> List[int]:
        """
        Correct solution, but not accepted.
        Leetcode time Limit exceeded; 35/48 testcases passed
        """
        n = len(temperatures)
        days_until_warmer = [0] * n
        i = 0
        while i < n:
            j = i + 1
            while j < n and temperatures[i] >= temperatures[j]:
                j += 1
            if i + 1 <= j < n:
                days_until_warmer[i] = j - i
            i += 1
        return days_until_warmer

    def solve_stack(self, temperatures: List[int]) -> List[int]:
        n = len(temperatures)
        days_until_warmer = [0] * n
        stack = []  # keep the indices of the elements in temperature array
        for i in range(n - 1, -1, -1):  # loop over in reverse order
            concurrent = temperatures[i]
            while stack and concurrent >= temperatures[stack[-1]]:
                # remove all elements with lower temperature
                # at later days from the stack until a higher temperature is found.
                # The higher temp index is not removed
                stack.pop()

            if stack:
                # delta between the idx of the current element and
                # the last element in stack (idx)
                days_until_warmer[i] = stack[-1] - i
            # add the current index to process next
            stack.append(i)
        return days_until_warmer

    def solve_stack_2(self, temperatures):
        days_until_warmer = [0] * len(temperatures)
        stack = []
        for i, temp in enumerate(temperatures):
            while stack and temperatures[stack[-1]] < temp:
                index = stack.pop()
                days_until_warmer[index] = i - index
            stack.append(i)

        return days_until_warmer
