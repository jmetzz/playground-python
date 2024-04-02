"""
https://leetcode.com/problems/isomorphic-strings/description/
205. Isomorphic Strings
Easy
Given two strings s and t, determine if they are isomorphic.

Two strings s and t are isomorphic if the characters in s can be replaced to get t.

All occurrences of a character must be replaced with another character while
preserving the order of characters. No two characters may map to the same character,
but a character may map to itself.



Example 1:
Input: s = "egg", t = "add"
Output: true

Example 2:
Input: s = "foo", t = "bar"
Output: false

Example 3:
Input: s = "paper", t = "title"
Output: true


Constraints:

1 <= s.length <= 5 * 104
t.length == s.length
s and t consist of any valid ascii character.
"""

from collections import defaultdict


def is_isomorphic_two_dicts(s: str, t: str) -> bool:
    if len(s) != len(t):
        return False
    s_map = {}
    reverse_map = {}  # since no letter can be mapped twice
    for i in range(len(s)):
        s_letter = s[i]
        t_letter = t[i]
        if (s_letter in s_map and s_map[s_letter] != t_letter) or (
            t_letter in reverse_map and reverse_map[t_letter] != s_letter
        ):
            return False
        else:
            s_map[s_letter] = t_letter
            reverse_map[t_letter] = s_letter

    return True


def is_isomorphic_one_dict(s: str, t: str) -> bool:
    letters_map = defaultdict()
    for s_letter, t_letter in zip(s, t):
        if s_letter in letters_map:
            if letters_map[s_letter] != t_letter:
                return False
            else:
                continue
        else:
            if t_letter in letters_map.values():
                return False
            letters_map[s_letter] = t_letter
            continue

    return True


def is_isomorphic_magic(s: str, t: str, debug=False) -> bool:
    """Determine if two strings, s and t, are isomorphic using a concise approach.

    It is a neat and fast solution (for leetcode leaderboard), but slightly difficult
    to understand why it works (for me anyways).

    Two strings are isomorphic if:
    - Each character in s can be replaced to get t.
    - All occurrences of a character must be replaced with another character
      while preserving the order of characters.
    - No two characters may map to the same character but a character may map to itself.

    To satisfy these isomorphic constraints, the function checks for:
    1. Each character mapping from `s` to `t` is unique.
    2. The total number of unique characters in `s` and `t` are equal.
    3. Each character in `s` maps to one and only one character in `t`, and vice versa.

    How it works:
    - By zipping `s` and `t`, we create pairs of corresponding characters.
    - Wrapping the `zip` result with a set removes any duplicate mappings.
    Let's keep this result in a set called `pairing`.
    - The isomorphic condition is satisfied if:
        a) The number of unique pairings equals the number of unique characters
        in `s`, ie. `len(pairing) == len(set(s))`.
        b) The number of unique characters in `s` equals the number of unique
        characters in `t`, ie. `len(set(s)) == len(set(t))`.
        These conditions together ensure a one-to-one mapping between
        characters of `s` and `t`.


    Expanding a bit...
    Note that zipping does not directly guarantee that a character in `s` will
    not be mapped to different characters in `t`.

    To guarantee that "No two characters may map to the same character", the
    length of the pairing set must be equal to the number of unique characters
    in `s`. If a character in `s` were mapped to multiple characters in `t`,
    there would be more unique characters in `s` than there are pairings,
    because duplicate pairings are removed by making pairing a set.


    Complexity:
        - Time complexity: $O(n)$, where `n` is the length of the strings `s` and `t`.
          The function iterates through each character in the strings exactly once
          to create the zipped pairs and the sets of characters from `s` and `t`.

        - Space complexity: $O(n)$, also linear because the function allocates
        additional space for the sets created from `s`, `t`, and the zipped pairs.

    Args:
        s (str): The first string.
        t (str): The second string, to be compared with s for isomorphism.
        debug (bool, optional): If True, prints the set of character pairings for debugging.
                                Default is False.

    Returns:
        bool: True if s and t are isomorphic; otherwise, False.
    """

    pairing = set(zip(s, t))

    if debug:
        print(pairing)

    return len(pairing) == len(set(s)) == len(set(t))
