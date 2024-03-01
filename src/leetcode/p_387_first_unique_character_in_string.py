"""
https://leetcode.com/problems/first-unique-character-in-a-sequence/description/
387. First Unique Character in a sequence


Easy
Given a sequence s, find the first non-repeating character in it and return its index.
If it does not exist, return -1.


Example 1:
Input: s = "leetcode"
Output: 0

Example 2:
Input: s = "loveleetcode"
Output: 2

Example 3:
Input: s = "aabb"
Output: -1


Constraints:
1 <= s.length <= 105
s consists of only lowercase English letters.
"""
import heapq
from collections import Counter
from math import inf


def first_unique_char_1(sequence: str) -> int:
    counter = Counter(sequence)
    unique_chars = {ch for ch, qtt in counter.items() if qtt == 1}
    n = len(sequence)
    first_idx = inf
    for ch in unique_chars:
        candidate_idx = sequence.index(ch)
        first_idx = min(first_idx, candidate_idx)
    return -1 if first_idx > n else first_idx


def first_unique_char_2(sequence: str) -> int:
    """
    Find the index of the first non-repeating character in the given sequence
    """
    counter = Counter(sequence)
    min_heap = [(sequence.index(ch), ch) for ch, count in counter.items() if count == 1]
    if not min_heap:
        # there is no unique character in the input sequence
        return -1
    heapq.heapify(min_heap)
    return min_heap[0][0]


def first_unique_char_3(sequence: str) -> int:
    """
    This solution will hit the "Time Limit Exceeded" in any coding platform.

    """
    for idx, e in enumerate(sequence):
        if sequence.count(e) == 1:
            return idx
    return -1


if __name__ == "__main__":
    inputs = ["leetcode", "loveleetcode", "aabb"]

    for s in inputs:
        print(first_unique_char_1(s))
        print(first_unique_char_2(s))
        print(first_unique_char_3(s))
        print("---")
