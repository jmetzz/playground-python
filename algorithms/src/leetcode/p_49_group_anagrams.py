"""
https://leetcode.com/problems/group-anagrams/description/

49. Group Anagrams
Medium

Given an array of strings strs, group the anagrams together. You can return the answer in any order.
An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase,
typically using all the original letters exactly once.



Example 1:
Input: strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]

Example 2:
Input: strs = [""]
Output: [[""]]

Example 3:
Input: strs = ["a"]
Output: [["a"]]


Constraints:
1 <= strs.length <= 104
0 <= strs[i].length <= 100
strs[i] consists of lowercase English letters
"""
from collections import defaultdict, Counter
from typing import List


class GroupAnagrams:
    def solve(self, strs: List[str]) -> List[List[str]]:
        anagrams_map = defaultdict(list)
        for word in strs:
            key = "".join(sorted(word))
            anagrams_map[key].append(word)
        return [group for _, group in anagrams_map.items()]

    def solve_with_custom_signature(self, strs: List[str]) -> List[List[str]]:
        anagrams_map = defaultdict(list)
        for word in strs:
            key = self._create_signature(word)
            anagrams_map[key].append(word)
        return [group for _, group in anagrams_map.items()]

    @classmethod
    def _create_signature(cls, s: str):
        # signature is made of the concatenation of char+count
        counter = Counter(s)
        keys = sorted([k for k in counter.keys()])
        return "".join([f"{k}{counter[k]}" for k in keys])
