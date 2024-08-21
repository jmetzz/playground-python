"""https://leetcode.com/problems/length-of-last-word/description

58. Length of Last Word
Easy

Given a string s consisting of words and spaces,
return the length of the last word in the string.

Definitions:
Substring: a substring is a contiguous non-empty sequence of characters
within a string.
Maximal substring: A word is a maximal substring consisting of
non-space characters only.


Example 1:
Input: s = "Hello World"
Output: 5
Explanation: The last word is "World" with length 5.

Example 2:
Input: s = "   fly me   to   the moon  "
Output: 4
Explanation: The last word is "moon" with length 4.

Example 3:
Input: s = "luffy is still joyboy"
Output: 6
Explanation: The last word is "joyboy" with length 6.


Constraints:

1 <= s.length <= 104
s consists of only English letters and spaces ' '.
There will be at least one word in s.
"""


def length_of_last_word_cheating_1(text: str) -> int:
    """Solve using built-in functions

    This function also applies side-effect on the input string.
    Don't do this in production code XD
    """
    text = text.rstrip()
    idx = text.rfind(" ")
    return len(text) if idx == -1 else len(text) - (idx + 1)


def length_of_last_word_cheating_2(text: str) -> int:
    """Solve using built-in functions"""
    words = text.split()
    if words:
        return len(words[-1])
    return 0


def length_of_last_word(text: str) -> int:
    """DYI solution"""
    if not text:
        return 0
    i = len(text) - 1
    # find the last valid char
    while text[i] == " " and i >= 0:
        i -= 1
    end_idx = i
    while text[i] != " " and i >= 0:
        i -= 1
    return end_idx - i
