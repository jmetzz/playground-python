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

from collections import Counter, defaultdict
from typing import List


def group_anagrams_solve(sequences: List[str]) -> List[List[str]]:
    anagrams_map = defaultdict(list)
    for word in sequences:
        key = "".join(sorted(word))
        anagrams_map[key].append(word)
    return list(anagrams_map.values())


def group_anagrams_solve_with_custom_signature(sequences: List[str]) -> List[List[str]]:
    anagrams_map = defaultdict(list)
    for word in sequences:
        key = _create_signature(word)
        anagrams_map[key].append(word)
    # return [group for group in anagrams_map.values()]
    return list(anagrams_map.values())


def _create_signature(sequence: str):
    # signature is made of the concatenation of char+count
    counter = Counter(sequence)
    keys = sorted([k for k in counter])
    return "".join([f"{k}{counter[k]}" for k in keys])
