"""
https://leetcode.com/problems/merge-intervals/description/
56. Merge Intervals
Medium

Given an array of intervals where intervals[i] = [starti, endi],
merge all overlapping intervals, and return an array of the non-overlapping
intervals that cover all the intervals in the input.



Example 1:
Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlap, merge them into [1,6].

Example 2:
Input: intervals = [[1,4],[4,5]]
Output: [[1,5]]
Explanation: Intervals [1,4] and [4,5] are considered overlapping.


Constraints:

1 <= intervals.length <= 104
intervals[i].length == 2
0 <= starti <= endi <= 104
"""

from typing import List


def merge_intervals(intervals: List[List[int]]) -> List[List[int]]:
    if len(intervals) == 1:
        return intervals[:]
    answer = []
    for _, curr in enumerate(sorted(intervals)):
        if not answer or answer[-1][1] < curr[0]:
            # either answer is empty or there is
            # non-overlaping between the last interval added
            # to answer, which represents the previous interval
            answer.append(curr)
        else:
            # there is an overlappving and we need to merge the intervals.
            # This is effectivelly done by expanding the interval
            answer[-1][1] = max(answer[-1][1], curr[1])
    return answer


def merge_intervals_2(intervals: List[List[int]]) -> List[List[int]]:
    if len(intervals) == 1:
        return intervals[:]
    intervals.sort(key=lambda x: x[0])
    answer = []
    for _, curr in enumerate(intervals):
        if not answer or answer[-1][1] < curr[0]:
            # either answer is empty or there is
            # non-overlaping between the last interval added
            # to answer, which represents the previous interval
            answer.append(curr)
        else:
            # there is an overlappving and we need to merge the intervals.
            # This is effectivelly done by expanding the interval
            answer[-1][1] = max(answer[-1][1], curr[1])
    return answer


if __name__ == "__main__":
    test_cases = [
        ([[0, 1], [4, 5], [6, 9]]),  # no overlap
        ([[4, 5], [5, 9], [12, 15]]),  # 2 overlapping on the boundary
        ([[1, 5], [3, 9]]),  # 2 overlapping
        ([[1, 2], [3, 4], [5, 6], [8, 10], [12, 16]]),  # no overlap
        ([[1, 2], [3, 6], [5, 6], [6, 9], [8, 10], [12, 16]]),  #  4 overlapping
    ]

    for intervals in test_cases:
        print(merge_intervals(intervals))
