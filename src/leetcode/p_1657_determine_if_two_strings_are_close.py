"""
https://leetcode.com/problems/determine-if-two-strings-are-close/description

1657. Determine if Two Strings Are Close
Solved
Medium
Topics
Companies
Hint
Two strings are considered close if you can attain one from the other using
the following operations:

Operation 1: Swap any two existing characters.
For example, abcde -> aecdb
Operation 2: Transform every occurrence of one existing character into another existing character,
and do the same with the other character.
For example, aacabb -> bbcbaa (all a's turn into b's, and all b's turn into a's)
You can use the operations on either string as many times as necessary.

Given two strings, word1 and word2, return true if word1 and word2 are close, and false otherwise.

Example 1:
Input: word1 = "abc", word2 = "bca"
Output: true
Explanation: You can attain word2 from word1 in 2 operations.
Apply Operation 1: "abc" -> "acb"
Apply Operation 1: "acb" -> "bca"

Example 2:
Input: word1 = "a", word2 = "aa"
Output: false
Explanation: It is impossible to attain word2 from word1, or vice versa,
in any number of operations.

Example 3:
Input: word1 = "cabbba", word2 = "abbccc"
Output: true
Explanation: You can attain word2 from word1 in 3 operations.
Apply Operation 1: "cabbba" -> "caabbb"
Apply Operation 2: "caabbb" -> "baaccc"
Apply Operation 2: "baaccc" -> "abbccc"


Constraints:
1 <= word1.length, word2.length <= 105
word1 and word2 contain only lowercase English letters.
"""
from collections import Counter


def close_strings_array(word1: str, word2: str) -> bool:
    if len(word1) != len(word2) or set(word1) != set(word2):
        return False

    hist_1 = [0] * 26
    hist_2 = [0] * 26
    for letter in word1:
        hist_1[ord(letter) - 97] += 1

    for letter in word2:
        hist_2[ord(letter) - 97] += 1

    for i in range(26):
        # ensure the presence of the same letters in both words
        if (hist_1[i] == 0 and hist_2[i] != 0) or (hist_1[i] != 0 and hist_2[i] == 0):
            return False

    return sorted(hist_1) == sorted(hist_2)


def close_strings_counter(word1: str, word2: str) -> bool:
    if len(word1) != len(word2) or set(word1) != set(word2):
        return False

    counter_1 = Counter(word1)
    counter_2 = Counter(word2)
    if counter_1.keys() != counter_2.keys():
        # ensure the presence of the same letters in both words
        return False

    hist_1 = sorted(counter_1.values())
    hist_2 = sorted(counter_2.values())
    return sorted(hist_1) == sorted(hist_2)


if __name__ == "__main__":
    inputs = [("abc", "bca"), ("a", "aa"), ("cabbba", "abbccc")]

    for str1, str2 in inputs:
        print(close_strings_array(str1, str2))
        print(close_strings_counter(str1, str2))
        print()
