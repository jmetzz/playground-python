"""https://leetcode.com/problems/valid-anagram/description

242. Valid Anagram
Easy

Given two strings s and t, return true if t is an anagram of s, and false otherwise.

An Anagram is a word or phrase formed by rearranging the letters of a different
word or phrase, typically using all the original letters exactly once.


Example 1:
Input: s = "anagram", t = "nagaram"
Output: true

Example 2:
Input: s = "rat", t = "car"
Output: false


Constraints:
1 <= s.length, t.length <= 5 * 104
s and t consist of lowercase English letters.


Follow up: What if the inputs contain Unicode characters?
How would you adapt your solution to such a case?
"""

from collections import Counter


def is_anagram_sorting(this: str, other: str) -> bool:
    return sorted(this) == sorted(other)


def is_anagram_counter(this: str, other: str) -> bool:
    if len(this) != len(other):
        return False

    counter = Counter(this)

    for ch in other:
        if ch not in counter:
            return False
        counter[ch] -= 1

    return sum(counter.values()) == 0


def is_anagram_256array(this: str, other: str) -> bool:
    if len(this) != len(other):
        return False

    chars = [0] * 256
    for ch in this:
        chars[ord(ch)] += 1
    for ch in other:
        chars[ord(ch)] -= 1

    return not any(chars)


def is_anagram_builtin_count(this: str, other: str) -> bool:
    if len(this) != len(other):
        return False
    return all(this.count(ch) == other.count(ch) for ch in set(other))


if __name__ == "__main__":
    print(is_anagram_sorting(this="anagram", other="nagaram"))
    print(is_anagram_sorting(this="rat", other="car"))

    print(is_anagram_counter(this="anagram", other="nagaram"))
    print(is_anagram_counter(this="rat", other="car"))

    print(is_anagram_256array(this="anagram", other="nagaram"))
    print(is_anagram_256array(this="rat", other="car"))

    print(is_anagram_builtin_count(this="anagram", other="nagaram"))
    print(is_anagram_builtin_count(this="rat", other="car"))
