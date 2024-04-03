"""
https://leetcode.com/problems/smallest-number-in-infinite-set/description
2336. Smallest Number in Infinite Set
Medium

You have a set which contains all positive integers [1, 2, 3, 4, 5, ...].

Implement the SmallestInfiniteSet class:

SmallestInfiniteSet() Initializes the SmallestInfiniteSet object to contain all positive integers.
int popSmallest() Removes and returns the smallest integer contained in the infinite set.
void addBack(int num) Adds a positive integer num back into the infinite set,
if it is not already in the infinite set.


Example 1:

Input
["SmallestInfiniteSet", "addBack", "popSmallest", "popSmallest", "popSmallest",
"addBack", "popSmallest", "popSmallest", "popSmallest"]
[[], [2], [], [], [], [1], [], [], []]
Output
[null, null, 1, 2, 3, null, 1, 4, 5]

Explanation
SmallestInfiniteSet smallestInfiniteSet = new SmallestInfiniteSet();
smallestInfiniteSet.addBack(2);    // 2 is already in the set, so no change is made.
smallestInfiniteSet.popSmallest(); // return 1, since 1 is the smallest number, and remove it from the set.
smallestInfiniteSet.popSmallest(); // return 2, and remove it from the set.
smallestInfiniteSet.popSmallest(); // return 3, and remove it from the set.
smallestInfiniteSet.addBack(1);    // 1 is added back to the set.
smallestInfiniteSet.popSmallest(); // return 1, since 1 was added back to the set and
                                   // is the smallest number, and remove it from the set.
smallestInfiniteSet.popSmallest(); // return 4, and remove it from the set.
smallestInfiniteSet.popSmallest(); // return 5, and remove it from the set.


Constraints:

1 <= num <= 1000
At most 1000 calls will be made in total to popSmallest and addBack.
"""

from heapq import heappop, heappush


class SmallestInfiniteSet_Naive:
    def __init__(self):
        self.regitry = [True] * 1000

    def popSmallest(self) -> int:
        for i, v in enumerate(self.regitry):
            if v:
                self.regitry[i] = False
                return i + 1

    def addBack(self, num: int) -> None:
        self.regitry[num - 1] = True


class SmallestInfiniteSet_Set_and_loop:
    """Handles the infinite set of positive integers as an implicit list

    by considering numbers that were already popped out, and therefore,
    are not available (not present) in the list.

    It uses set operations which are O(1) for presence check and removal.
    """

    def __init__(self, max_value: int = 1000):
        self.unavailable = set()
        self.max_value = max_value

    def popSmallest(self) -> int:
        """Iterate over the unavailable set
        to check what is the smallest value not yet
        in the set, which is the smalles values
        available for the pop operation"""
        num = 1
        while num in self.unavailable:
            num += 1

        self.unavailable.add(num)
        return num

    def addBack(self, num: int) -> None:
        """Remove num from the unavailable set,
        which represents adding it back into the
        set of available numbers to be popped"""
        # Use discard function to remove num from the set if it is present,
        # to avoid having to handle KeyError in case it is not present.
        # This exception is raised by the function remove in the element
        # is not included in the set.
        self.unavailable.discard(num)


class SmallestInfiniteSet_Heap:
    def __init__(self):
        self.smalles_available = 1
        self.added_back = set()
        self.added_min_heap = []

    def popSmallest(self) -> int:
        if self.added_back:
            num = heappop(self.added_min_heap)
            self.added_back.remove(num)
        else:
            num = self.smalles_available
            self.smalles_available += 1
        return num

    def addBack(self, num: int) -> None:
        if num >= self.smalles_available or num in self.added_back:
            return
        heappush(self.added_min_heap, num)
        self.added_back.add(num)


class SmallestInfiniteSet_Set:
    def __init__(self):
        self.smalles_available = 1
        self.added_back = set()

    def popSmallest(self) -> int:
        if self.added_back:
            min_val = min(self.added_back)
            self.added_back.remove(min_val)
            return min_val
        else:
            num = self.smalles_available
            self.smalles_available += 1
            return num

    def addBack(self, num: int) -> None:
        if num < self.smalles_available:
            self.added_back.add(num)
