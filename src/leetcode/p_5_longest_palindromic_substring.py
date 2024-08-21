"""5. Longest Palindromic Substring
Medium

Given a string s, return the longest palindromic substring in s.

Example 1:
Input: s = "babad"
Output: "bab"
Explanation: "aba" is also a valid answer.

Example 2:
Input: s = "cbbd"
Output: "bb"


Constraints:
1 <= s.length <= 1000
s consist of only digits and English letters.
"""

import heapq


def longest_palindrome_brute_force(s: str) -> str:
    answer = s[0]
    n = len(s)
    for size in range(n, 1, -1):
        for i in range(n - size, -1, -1):
            ss = s[i : i + size]
            if ss == ss[::-1]:
                answer = ss if len(ss) > len(answer) else answer
    return answer


def longest_palindrome_heap_brute_force(s: str) -> str:
    substrings = set(s)
    n = len(s)
    for size in range(2, n + 1):
        for i in range(n - size, -1, -1):
            ss = s[i : i + size]
            if ss == ss[::-1]:
                substrings.add(s[i : i + size])
    h = [(-len(ss), ss) for ss in substrings]
    heapq.heapify(h)
    _, answer = h[0]
    return answer


def longest_palindrome_expanding_window(sequence: str) -> str:
    """To optimize the execution we can reuse a previously computed palindrome
    to compute a larger palindrome.
    If “aba” is a palindrome, “XabaX” is also a palindrome.
    Also, if “XabaZ” is not a palindrome, the other characters added to left and right
    will not make the new string a palindrome.
    """
    if sequence[::-1] == sequence:
        return sequence

    answer = ""
    for mid in range(len(sequence)):
        # check odd length palindromes
        even_length = backtrack_palindrome(sequence, mid, mid)

        # check even length palindromes
        odd_length = backtrack_palindrome(sequence, mid, mid + 1)
        candidate = even_length if len(even_length) > len(odd_length) else odd_length
        if len(candidate) > len(answer):
            answer = candidate
    return answer


def backtrack_palindrome(sequence: str, left: int, right: int) -> str:
    # expand the window while it is a palindrome
    while left >= 0 and right < len(sequence) and sequence[left] == sequence[right]:
        left, right = left - 1, right + 1
    # return the string in the window
    return sequence[left + 1 : right]


if __name__ == "__main__":
    inputs = [
        # "cbbd",
        # "babad",
        "abbbc"
    ]

    for word in inputs:
        # print(longest_palindrome_brute_force(word))
        # print(longest_palindrome_heap_brute_force(word))
        # print(longest_palindrome_expanding_window(word))
        for i in range(len(word)):
            print(backtrack_palindrome(word, i, i))
        print()
