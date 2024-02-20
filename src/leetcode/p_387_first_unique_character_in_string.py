"""
https://leetcode.com/problems/first-unique-character-in-a-string/description/
387. First Unique Character in a String


Easy
Given a string s, find the first non-repeating character in it and return its index.
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
from collections import Counter


class FirstUniqChar:
    def solve(self, s: str) -> int:
        counter = Counter(s)
        unique_chars = {c for c, qtt in counter.items() if qtt == 1}
        n = len(s)
        first_idx = n + 1
        for c in unique_chars:
            p = s.index(c)
            first_idx = min(first_idx, p)
        return -1 if first_idx > n else first_idx
