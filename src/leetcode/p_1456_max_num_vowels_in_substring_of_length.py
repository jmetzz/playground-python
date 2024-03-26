"""
https://leetcode.com/problems/maximum-number-of-vowels-in-a-substring-of-given-length/description/

456. Maximum Number of Vowels in a Substring of Given Length
Medium

Given a string s and an integer k, return the maximum number of vowel letters in any substring of s with length k.

Vowel letters in English are 'a', 'e', 'i', 'o', and 'u'.



Example 1:
Input: s = "abciiidef", k = 3
Output: 3
Explanation: The substring "iii" contains 3 vowel letters.

Example 2:
Input: s = "aeiou", k = 2
Output: 2
Explanation: Any substring of length 2 contains 2 vowels.

Example 3:
Input: s = "leetcode", k = 3
Output: 2
Explanation: "lee", "eet" and "ode" contain 2 vowels.


Constraints:
1 <= s.length <= 105
s consists of lowercase English letters.
1 <= k <= s.length
"""

VOWELS = "aeiou"


def max_vowels(sequence: str, window_size: int) -> int:
    max_counter = 0
    left, right = 0, 0
    while right < window_size:
        if sequence[right] in VOWELS:
            max_counter += 1
        right += 1
    previous = max_counter
    while right < len(sequence):
        # we can move the window to the right 1 slot,
        # discounting and adding vowels as we move the window
        if sequence[left] in VOWELS:
            previous -= 1

        if sequence[right] in VOWELS:
            previous += 1
        left += 1
        right += 1
        max_counter = max(max_counter, previous)
    return max_counter


if __name__ == "__main__":
    print(max_vowels("abciiidef", 3))
    print(max_vowels("aeiou", 2))
    print(max_vowels("leetcode", 3))
