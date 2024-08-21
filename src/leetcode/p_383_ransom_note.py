"""https://leetcode.com/problems/ransom-note/description/
383. Ransom Note
Easy

Given two strings ransomNote and magazine,
return true if ransomNote can be constructed by using
the letters from magazine and false otherwise.

Each letter in magazine can only be used once in ransomNote.



Example 1:

Input: ransomNote = "a", magazine = "b"
Output: false

Example 2:
Input: ransomNote = "aa", magazine = "ab"
Output: false

Example 3:
Input: ransomNote = "aa", magazine = "aab"
Output: true

Constraints:
1 <= ransomNote.length, magazine.length <= 105
ransomNote and magazine consist of lowercase English letters.


"""

from collections import Counter


def can_construct_1(ransom_note: str, magazine: str) -> bool:
    required_chars_counter = Counter(ransom_note)
    available_chars = Counter(magazine)
    for ch, count in required_chars_counter.items():
        if ch not in available_chars or available_chars[ch] < count:
            return False

    return True


def can_construct_2(ransom_note: str, magazine: str) -> bool:
    available_chars = Counter(magazine)
    for ch in ransom_note:
        if ch not in available_chars or available_chars[ch] == 0:
            return False
        available_chars[ch] -= 1
    return True


def can_construct_3(ransom_note: str, magazine: str) -> bool:
    for ch in set(ransom_note):
        if ransom_note.count(ch) > magazine.count(ch):
            return False

    # A note about the '# '
    # this function could be simplified (in terms of readability) as
    # return all(ransom_note.count(ch) <= magazine.count(ch) for ch in set(ransom_note))

    return True
