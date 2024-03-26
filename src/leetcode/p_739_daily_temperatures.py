"""
https://leetcode.com/problems/daily-temperatures/description/

739. Daily Temperatures

Medium


Given an array of integers temperatures represents the daily temperatures,
return an array answer such that answer[i] is the number of days
you have to wait after the ith day to get a warmer temperature.
If there is no future day for which this is possible, keep answer[i] == 0 instead.



Example 1:
Input: temperatures = [73,74,75,71,69,72,76,73]
Output: [1,1,4,2,1,1,0,0]

Example 2:
Input: temperatures = [30,40,50,60]
Output: [1,1,1,0]

Example 3:
Input: temperatures = [30,60,90]
Output: [1,1,0]


Constraints:
1 <= temperatures.length <= 10^5
30 <= temperatures[i] <= 100
"""

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
