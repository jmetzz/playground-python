"""
https://leetcode.com/problems/trapping-rain-water/description
42. Trapping Rain Water
Hard

Topics: #Array #Two-Pointers #Dynamic-Programming #Stack #Monotonic-Stack

Given n non-negative integers representing an elevation map where the width of each bar is 1,
compute how much water it can trap after raining.



Example 1:
Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
Explanation: The above elevation map (black section) is represented by array
[0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6 units of rain water (blue section) are being trapped.

Example 2:
Input: height = [4,2,0,3,2,5]
Output: 9


Constraints:

n == height.length
1 <= n <= 2 * 104
0 <= height[i] <= 105
"""

from typing import List


def trapped_water_volume_two_pointers(height: List[int]) -> int:
    n = len(height)
    if n == 0:
        return 0

    left_heights, right_heights = [0] * n, [0] * n

    left_heights[0] = height[0]
    for idx in range(1, n):
        left_heights[idx] = max(left_heights[idx - 1], height[idx])

    right_heights[-1] = height[-1]
    for idx in reversed(range(n - 1)):
        right_heights[idx] = max(right_heights[idx + 1], height[idx])

    total_volume = 0
    for i in range(n):
        total_volume += min(left_heights[i], right_heights[i]) - height[i]
    return total_volume


def trapped_water_volume_monotonic_stack(heights: List[int]) -> int:
    """
    Use a stack to maintain a list of indices with bars where
    the next bars are smaller or equal in height

    """
    n = len(heights)
    stack = []
    total_water_trapped = 0
    for current in range(n):
        while stack and heights[stack[-1]] < heights[current]:
            valley_index = stack.pop()

            if not stack:
                break

            # The current element and the element corresponding to
            # the new top of the stack form the boundaries
            left_bound_index = stack[-1]
            right_bound_index = current

            # Width between the left and right boundary
            width = right_bound_index - left_bound_index - 1

            # Height is determined by the shorter of the two boundaries minus the height of the valley
            h = min(heights[left_bound_index], heights[right_bound_index]) - heights[valley_index]

            # Calculate trapped water and add to the total
            total_water_trapped += width * h

        stack.append(current)
    return total_water_trapped
