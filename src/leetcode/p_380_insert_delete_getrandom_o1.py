"""
https://leetcode.com/problems/insert-delete-getrandom-o1/description

380. Insert Delete GetRandom O(1)
Medium

Implement the RandomizedSet class:

RandomizedSet() Initializes the RandomizedSet object.
bool insert(int val) Inserts an item val into the set if not present.
Returns true if the item was not present, false otherwise.
bool remove(int val) Removes an item val from the set if present.
Returns true if the item was present, false otherwise.
int getRandom() Returns a random element from the current set of elements
(it's guaranteed that at least one element exists when this method is called).
Each element must have the same probability of being returned.
You must implement the functions of the class such that each function
works in average O(1) time complexity.


Example 1:
Input
["RandomizedSet", "insert", "remove", "insert", "getRandom", "remove", "insert", "getRandom"]
[[], [1], [2], [2], [], [1], [2], []]
Output
[null, true, false, true, 2, true, false, 2]

Explanation
RandomizedSet randomizedSet = new RandomizedSet();
randomizedSet.insert(1); // Inserts 1 to the set. Returns true as 1 was inserted successfully.
randomizedSet.remove(2); // Returns false as 2 does not exist in the set.
randomizedSet.insert(2); // Inserts 2 to the set, returns true. Set now contains [1,2].
randomizedSet.getRandom(); // getRandom() should return either 1 or 2 randomly.
randomizedSet.remove(1); // Removes 1 from the set, returns true. Set now contains [2].
randomizedSet.insert(2); // 2 was already in the set, so return false.
randomizedSet.getRandom(); // Since 2 is the only number in the set, getRandom() will always return 2.


Constraints:

-2^31 <= val <= 2^31 - 1
At most 2 * 10^5 calls will be made to insert, remove, and getRandom.
There will be at least one element in the data structure when getRandom is called.
"""

import random


class RandomizedSet:
    def __init__(self):
        self._elements = set()

    def insert(self, val: int) -> bool:
        if val in self._elements:
            return False
        self._elements.add(val)
        return True

    def remove(self, val: int) -> bool:
        was_present = val in self._elements
        self._elements.discard(val)
        return was_present

    def getRandom(self) -> int:
        item = random.randint(0, len(self._elements) - 1)
        return list(self._elements)[item]


class RandomizedSetBetter:
    def __init__(self):
        self._elements = []
        self._idx_map = {}  # unordered map

    def insert(self, val: int) -> bool:
        if val in self._idx_map:
            return False
        self._elements.append(val)
        self._idx_map[val] = len(self._elements) - 1
        return True

    def remove(self, val: int) -> bool:
        if val not in self._idx_map:
            return False

        idx = self._idx_map[val]
        # move the last element to the place of the target element
        self._elements[idx] = self._elements[-1]
        # adjust the index in the mapping
        self._idx_map[self._elements[idx]] = idx
        # and remover the element from the collections
        self._elements.pop()
        del self._idx_map[val]
        return True

    def getRandom(self) -> int:
        return random.choice(self._elements)


if __name__ == "__main__":
    obj = RandomizedSetBetter()
    output = []
    output.append(obj.insert(1))
    output.append(obj.remove(2))
    output.append(obj.insert(2))
    output.append(obj.getRandom())
    output.append(obj.remove(1))
    output.append(obj.insert(2))
    output.append(obj.getRandom())
    print(obj._elements)
    print(output)

    obj = RandomizedSetBetter()
    output = []
    output.append(obj.insert(1))
    output.append(obj.insert(10))
    output.append(obj.insert(20))
    output.append(obj.insert(30))
    output.append(obj.getRandom())
    output.append(obj.getRandom())
    output.append(obj.getRandom())
    output.append(obj.getRandom())
    output.append(obj.getRandom())
    output.append(obj.getRandom())
    print(output)
    print(obj._elements)
