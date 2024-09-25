"""
https://leetcode.com/problems/lru-cache/description

146. LRU Cache

Medium

Design a data structure that follows the constraints of a
Least Recently Used (LRU) cache.
<https://en.wikipedia.org/wiki/Cache_replacement_policies#LRU>

Implement the LRUCache class:

* LRUCache(int capacity) Initialize the LRU cache with positive size capacity.
* int get(int key) Return the value of the key if the key exists,
otherwise return -1.
* void put(int key, int value) Update the value of the key if the key exists.
Otherwise, add the key-value pair to the cache. If the number of keys exceeds
the capacity from this operation, evict the least recently used key.

The functions get and put must each run in O(1) average time complexity.


Example 1:

Input
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
Output
[null, null, null, 1, null, -1, null, -1, 3, 4]

Explanation
LRUCache lRUCache = new LRUCache(2);
lRUCache.put(1, 1); // cache is {1=1}
lRUCache.put(2, 2); // cache is {1=1, 2=2}
lRUCache.get(1);    // return 1
lRUCache.put(3, 3); // LRU key was 2, evicts key 2, cache is {1=1, 3=3}
lRUCache.get(2);    // returns -1 (not found)
lRUCache.put(4, 4); // LRU key was 1, evicts key 1, cache is {4=4, 3=3}
lRUCache.get(1);    // return -1 (not found)
lRUCache.get(3);    // return 3
lRUCache.get(4);    // return 4


Constraints:

1 <= capacity <= 3000
0 <= key <= 10^4
0 <= value <= 10^5
At most 2 * 10^5 calls will be made to get and put.
"""

from collections import OrderedDict


class Node:
    def __init__(self, key: int, val: int) -> None:
        self.key, self.val = key, val
        self.prev = None
        self.next = None

    def __repr__(self) -> str:
        prev_key = self.prev.key if self.prev else None
        next_key = self.next.key if self.next else None
        return f"Node(key={self.key}, val={self.val}, prev={prev_key}, next={next_key})"

    def __str__(self) -> str:
        return f"Node({self.key}: {self.val})"


class LRUCache:
    def __init__(self, capacity: int):
        self._capacity = capacity
        self._lru_cache = {}  # map the key to Node instances
        self.dummy_head, self.dummy_tail = Node(None, None), Node(None, None)  # dummy nodes
        self.dummy_head.next = self.dummy_tail
        self.dummy_tail.prev = self.dummy_head

    def _remove_node(self, node: Node):
        prev, nxt = node.prev, node.next
        prev.next = nxt
        nxt.prev = prev

    def _insert_node(self, node: Node):
        # insert at right position (the most recently used) ...
        # remember self.right is a dummy node is not part of the elements
        prev = self.dummy_tail.prev

        prev.next = node
        node.prev = prev

        node.next = self.dummy_tail
        self.dummy_tail.prev = node

    def get(self, key: int) -> int:
        node = self._lru_cache.get(key)
        if node:
            # promote to most recently used
            self._remove_node(node)
            self._insert_node(node)
            # then return the value
            return node.val
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self._lru_cache:
            node = self._lru_cache[key]
            node.val = value  # Update the value in the existing node
            self._remove_node(self._lru_cache[key])
        else:
            node = Node(key, value)
            self._lru_cache[key] = node

        self._insert_node(node)

        # evict the least recently used element, if necessary
        if len(self._lru_cache) > self._capacity:
            lru = self.dummy_head.next
            self._remove_node(lru)
            del self._lru_cache[lru.key]

    def __str__(self) -> str:
        nodes = []
        current = self.dummy_head.next
        while current != self.dummy_tail:
            nodes.append(str(current))
            current = current.next
        return " -> ".join(nodes)

    def __repr__(self) -> str:
        nodes = []
        current = self.left.next  # Start after the dummy head
        while current != self.right:
            nodes.append(f"{current.key}: {current.val}")
            current = current.next
        return f"LRUCache({', '.join(nodes)})"


class LRUCacheOrderedDict:
    def __init__(self, capacity: int):
        self._capacity = capacity
        self._lru_cache = OrderedDict()

    def get(self, key: int) -> int:
        if key in self._lru_cache:
            value = self._lru_cache[key]
            self._lru_cache.move_to_end(key)
            return value
        return -1

    def put(self, key: int, value: int) -> None:
        self._lru_cache[key] = value
        self._lru_cache.move_to_end(key)

        if len(self._lru_cache) > self._capacity:
            self._lru_cache.popitem(last=False)


if __name__ == "__main__":
    # Initialize the LRU cache with a capacity of 2
    lru_cache = LRUCache(2)

    print("Initial cache state: []")

    # Perform some operations and print the result after each one
    print("\nPut (1, 1)")
    lru_cache.put(1, 1)  # Cache is {1=1}
    print("Cache: ", lru_cache)

    print("\nPut (2, 2)")
    lru_cache.put(2, 2)  # Cache is {1=1, 2=2}
    print("Cache: ", lru_cache)

    print("\nGet key 1")
    print("Result: ", lru_cache.get(1))  # Returns 1, cache is {2=2, 1=1}
    print("Cache: ", lru_cache)

    print("\nPut (3, 3) - Evict key 2")
    lru_cache.put(3, 3)  # Cache evicts key 2, now {1=1, 3=3}
    print("Cache: ", lru_cache)

    print("\nGet key 2")
    print("Result: ", lru_cache.get(2))  # Returns -1 (not found)
    print("Cache: ", lru_cache)

    print("\nPut (4, 4) - Evict key 1")
    lru_cache.put(4, 4)  # Cache evicts key 1, now {4=4, 3=3}
    print("Cache: ", lru_cache)

    print("\nGet key 1")
    print("Result: ", lru_cache.get(1))  # Returns -1 (not found)
    print("Cache: ", lru_cache)

    print("\nGet key 3")
    print("Result: ", lru_cache.get(3))  # Returns 3
    print("Cache: ", lru_cache)

    print("\nGet key 4")
    print("Result: ", lru_cache.get(4))  # Returns 4
    print("Cache: ", lru_cache)
