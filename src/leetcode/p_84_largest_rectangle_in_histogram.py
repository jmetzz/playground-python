"""
https://leetcode.com/problems/largest-rectangle-in-histogram/description/

84. Largest Rectangle in Histogram
Hard

Given an array of integers heights representing the histogram's bar height where
the width of each bar is 1, return the area of the largest rectangle in the histogram.



Example 1:
Input: heights = [2,1,5,6,2,3]
Output: 10
Explanation: The above is a histogram where width of each bar is 1.
The largest rectangle is shown in the red area, which has an area = 10 units.

Example 2:
Input: heights = [2,4]
Output: 4

Explanation: https://youtu.be/zx5Sw9130L0?si=8Q7_AY9_ui249f47

Constraints:

1 <= heights.length <= 105
0 <= heights[i] <= 104

"""

from typing import List


def largest_rectangle_area(heights: List[int]) -> int:
    # use a monotonic increasing stack. If the element to add
    # to the stack would decrease top value, we need to remove
    # values until we can add the new element
    stack = []  # keep pairs (height, index), where index represent the stating index of the rectangle with this height
    n = len(heights)
    max_area = 0
    for idx, h in enumerate(heights):
        start = idx
        while stack and stack[-1][0] > h:
            prev_h, prev_start_idx = stack.pop()
            max_area = max(max_area, prev_h * (idx - prev_start_idx))
            start = prev_start_idx

        stack.append((h, start))

    while stack:
        h, idx = stack.pop()
        max_area = max(max_area, h * (n - idx))

    return max_area
