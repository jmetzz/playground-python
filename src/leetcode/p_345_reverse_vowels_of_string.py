"""
https://leetcode.com/problems/reverse-vowels-of-a-string/description
345. Reverse Vowels of a String
Easy

Given a string s, reverse only all the vowels in the string and return it.

The vowels are 'a', 'e', 'i', 'o', and 'u', and they can appear in both lower and upper cases, more than once.



Example 1:
Input: s = "hello"
Output: "holle"

Example 2:
Input: s = "leetcode"
Output: "leotcede"


Constraints:
1 <= s.length <= 3 * 105
s consist of printable ASCII characters.
"""
from collections import deque


def reverse_vowels(sequence: str) -> str:
    left = 0
    right = len(sequence) - 1
    vowels = "aeiouAEIOU"
    chars = list(sequence)
    while left < right:
        # advance left until the next vowel
        while left < right and chars[left] not in vowels:
            left += 1

        # move right to the left until the next vowel
        while right > left and chars[right] not in vowels:
            right -= 1

        # swap the vowels
        if left < right:
            chars[left], chars[right] = chars[right], chars[left]
        left += 1
        right -= 1
    return "".join(chars)


def reverse_vowels_stack(sequence: str) -> str:
    vowels = "aeiouAEIOU"
    chars = list(sequence)
    d = deque([(ch, idx) for idx, ch in enumerate(chars) if ch in vowels])
    if len(d) <= 1:
        return sequence
    while len(d) > 1:
        right, j = d.popleft()
        left, i = d.pop()
        chars[j] = left
        chars[i] = right
    return "".join(chars)


def reverse_vowels_2(sequence: str) -> str:
    vowels = "aeiouAEIOU"
    chars, indices = [], []
    for i, char in enumerate(sequence):
        if char in vowels:
            chars.append(char)
            indices.append(i)

    chars.reverse()
    answer = list(sequence)
    for i, ch in zip(indices, chars):
        answer[i] = ch
    return "".join(answer)


if __name__ == "__main__":
    inputs = ["hello", "leetcode", "race car"]

    for word in inputs:
        print(reverse_vowels(word))
        print(reverse_vowels_2(word))
        print(reverse_vowels_stack(word))
        print()
