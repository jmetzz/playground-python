"""https://leetcode.com/problems/insert-interval/description/

57. Insert Interval
Medium

You are given an array of non-overlapping intervals intervals where intervals[i] = [starti, endi]
represent the start and the end of the ith interval and intervals is sorted in ascending order by starti.
You are also given an interval newInterval = [start, end] that represents the start and end of another interval.

Insert newInterval into intervals such that intervals is still sorted in ascending order by starti and
intervals still does not have any overlapping intervals (merge overlapping intervals if necessary).

Return intervals after the insertion.
Note that you don't need to modify intervals in-place. You can make a new array and return it.


Example 1:
Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
Output: [[1,5],[6,9]]

Example 2:
Input: intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
Output: [[1,2],[3,10],[12,16]]
Explanation: Because the new interval [4,8] overlaps with [3,5],[6,7],[8,10].


Constraints:

0 <= intervals.length <= 104
intervals[i].length == 2
0 <= starti <= endi <= 105
intervals is sorted by starti in ascending order.
newInterval.length == 2
0 <= start <= end <= 105
"""


def insert_interval(intervals: list[list[int]], new_interval: list[int]) -> list[list[int]]:
    # base cases
    if not intervals:
        return [new_interval]
    if new_interval[1] < intervals[0][0]:
        # new interval should be the first element
        return [new_interval] + intervals
    if new_interval[0] > intervals[-1][1]:
        # new interval should be the last element
        return intervals + [new_interval]

    answer = []
    merged_interval = new_interval.copy()
    for i, interval in enumerate(intervals):
        if interval[1] < merged_interval[0]:
            # current interval ends before new_interval starts
            answer.append(interval)
        elif interval[0] > merged_interval[1]:
            # current interval starts after new_interval ends
            answer.append(merged_interval)
            return answer + intervals[i:]
        else:
            # overlapping case: merge the current interval
            merged_interval = [min(merged_interval[0], interval[0]), max(merged_interval[1], interval[1])]
    return answer + [merged_interval]


def insert_interval_2(intervals: list[list[int]], new_interval: list[int]) -> list[list[int]]:
    if not intervals:
        return [new_interval]
    left_elements = []
    right_elements = []
    merged_interval = new_interval

    for interval in intervals:
        if interval[1] < new_interval[0]:
            left_elements.append(interval)
        elif interval[0] > new_interval[1]:
            right_elements.append(interval)
        else:
            merged_interval = [min(merged_interval[0], interval[0]), max(merged_interval[1], interval[1])]
    return left_elements + [merged_interval] + right_elements


def insert_interval_3(intervals: list[list[int]], new_interval: list[int]) -> list[list[int]]:
    if not intervals:
        return [new_interval]

    left_elements = []
    right_elements = []
    merged = set()

    for interval in intervals:
        if interval[1] < new_interval[0]:
            left_elements.append(interval)
        elif interval[0] > new_interval[1]:
            right_elements.append(interval)
        else:
            merged.update(interval)
    if not merged:
        return left_elements + [new_interval] + right_elements

    merged.update(new_interval)
    merged = sorted(list(merged))
    return left_elements + [[merged[0], merged[-1]]] + right_elements
