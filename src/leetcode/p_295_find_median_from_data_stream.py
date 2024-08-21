"""https://leetcode.com/problems/find-median-from-data-stream/description/

295. Find Median from Data Stream
Hard

The median is the middle value in an ordered integer list. If the size of the list is even,
there is no middle value, and the median is the mean of the two middle values.

For example, for arr = [2,3,4], the median is 3.
For example, for arr = [2,3], the median is (2 + 3) / 2 = 2.5.
Implement the MedianFinder class:

MedianFinder() initializes the MedianFinder object.
void addNum(int num) adds the integer num from the data stream to the data structure.
double findMedian() returns the median of all elements so far. Answers within 10-5 of the actual
answer will be accepted.


Example 1:
Input
["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
[[], [1], [2], [], [3], []]
Output
[null, null, null, 1.5, null, 2.0]

Explanation
MedianFinder medianFinder = new MedianFinder();
medianFinder.addNum(1);    // arr = [1]
medianFinder.addNum(2);    // arr = [1, 2]
medianFinder.findMedian(); // return 1.5 (i.e., (1 + 2) / 2)
medianFinder.addNum(3);    // arr[1, 2, 3]
medianFinder.findMedian(); // return 2.0


Constraints:
-105 <= num <= 105
There will be at least one element in the data structure before calling findMedian.
At most 5 * 104 calls will be made to addNum and findMedian.


Follow up:
If all integer numbers from the stream are in the range [0, 100], how would you optimize your solution?
If 99% of all integer numbers from the stream are in the range [0, 100], how would you optimize your solution?
"""

from heapq import heappop, heappush


class MedianFinder:
    def __init__(self):
        self.left = []  # max heap
        self.right = []  # min heap

    def addNum(self, num: int) -> None:
        # always add to the left heap by default,
        heappush(self.left, -1 * num)

        # ensure ordering condition:
        # all elements in left must be <= than all elements in right
        if self.left and self.right and (-self.left[0] > self.right[0]):
            value = -heappop(self.left)
            heappush(self.right, value)

        # ensure length condition:
        # both heaps must be approximately the same
        # max 1 element difference
        if len(self.left) > len(self.right) + 1:
            heappush(self.right, -heappop(self.left))

        if len(self.right) > len(self.left) + 1:
            heappush(self.left, -heappop(self.right))

    def findMedian(self) -> float:
        if len(self.left) == len(self.right):
            return (-self.left[0] + self.right[0]) / 2
        return self.right[0] if len(self.left) < len(self.right) else -self.left[0]


if __name__ == "__main__":
    # Your MedianFinder object will be instantiated and called as such:
    obj = MedianFinder()
    obj.addNum(1)
    obj.addNum(2)
    print(obj.findMedian())
    obj.addNum(3)
    print(obj.findMedian())
