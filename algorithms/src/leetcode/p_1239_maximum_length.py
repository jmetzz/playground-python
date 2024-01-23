"""
https://leetcode.com/problems/maximum-length-of-a-concatenated-string-with-unique-characters

1239. Maximum Length of a Concatenated String with Unique Characters
Medium
Topics
Companies
Hint
You are given an array of strings arr. A string s is formed by the concatenation of a
subsequence of arr that has unique characters.

Return the maximum possible length of s.

A subsequence is an array that can be derived from another array by deleting some or
no elements without changing the order of the remaining elements.


Example 1:

Input: arr = ["un","iq","ue"]
Output: 4
Explanation: All the valid concatenations are:
- ""
- "un"
- "iq"
- "ue"
- "uniq" ("un" + "iq")
- "ique" ("iq" + "ue")
Maximum length is 4.
Example 2:

Input: arr = ["cha","r","act","ers"]
Output: 6
Explanation: Possible longest valid concatenations are "chaers" ("cha" + "ers") and "acters" ("act" + "ers").
Example 3:

Input: arr = ["abcdefghijklmnopqrstuvwxyz"]
Output: 26
Explanation: The only string in arr has all 26 characters.


Constraints:

1 <= arr.length <= 16
1 <= arr[i].length <= 26
arr[i] contains only lowercase English letters.
"""
from typing import List


class MaxLengthStrWithUniqueChars:
    def solve(self, arr: List[str]) -> int:
        max_len = [0]
        self._back_track(self._clean_up(arr), "", 0, max_len)
        return max_len[0]

    def _back_track(self, arr, current, start, max_len):
        if max_len[0] < len(current):
            max_len[0] = len(current)

        for i in range(start, len(arr)):
            if self._invalid(current, arr[i]):
                continue

            self._back_track(arr, current + arr[i], i + 1, max_len)

    def _clean_up(self, arr: List) -> List:
        return [v for v in arr if len(set(v)) == len(v)]

    def _invalid(self, current, new_string):
        return len(set(current) & set(new_string)) != 0


if __name__ == '__main__':
    print(MaxLengthStrWithUniqueChars().solve(["un", "iq", "ue"]))
    print(MaxLengthStrWithUniqueChars().solve(["cha", "r", "act", "ers"]))
