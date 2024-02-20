"""
https://leetcode.com/problems/palindromic-substrings/description/

647. Palindromic Substrings
Medium
Given a string s, return the number of palindromic substrings in it.

A string is a palindrome when it reads the same backward as forward.

A substring is a contiguous sequence of characters within the string.



Example 1:
Input: s = "abc"

Output: 3
Explanation: Three palindromic strings: "a", "b", "c".

Example 2:
Input: s = "aaa"
Output: 6

Explanation: Six palindromic strings: "a", "a", "a", "aa", "aa", "aaa".

Constraints:
1 <= s.length <= 1000
s consists of lowercase English letters.
"""


class PalindromicSubstrings:
    @staticmethod
    def is_palindrome(sequence: str) -> bool:
        n = len(sequence)
        if n == 1:
            return True

        right = n // 2
        left = right - 1 if n % 2 == 0 else right
        while left >= 0 and right < len(sequence) and sequence[left] == sequence[right]:
            left, right = left - 1, right + 1

        return right >= len(sequence)

    def count_palindromic_substrings_loop(self, sequence: str) -> int:
        n = len(sequence)
        count = 0
        for i in range(n):
            for j in range(i + 1, n + 1):
                if self.is_palindrome(sequence[i:j]):
                    count += 1

        return count

    def count_palindromic_substr_expanding_window(self, sequence: str) -> int:
        count = 0
        for idx in range(len(sequence)):
            # count all polindrome subsequence of ODD size from idx
            left, right = idx, idx
            while left >= 0 and right < len(sequence) and sequence[left] == sequence[right]:
                left, right = left - 1, right + 1
                count += 1

            # count all polindrome subsequence of EVEN size from given indices
            left, right = idx, idx + 1
            while left >= 0 and right < len(sequence) and sequence[left] == sequence[right]:
                left, right = left - 1, right + 1
                count += 1
        return count


if __name__ == "__main__":
    print("==== Is palindrome?")
    print(f"aaaa: {PalindromicSubstrings.is_palindrome("aaaa")}")
    print(f"ababac: {PalindromicSubstrings.is_palindrome("ababac")}")
    print(f"Hello: {PalindromicSubstrings.is_palindrome("Hello")}")

    print("==== Solution: loop")
    print(PalindromicSubstrings().count_palindromic_substrings_loop("a"))  # 1
    print(PalindromicSubstrings().count_palindromic_substrings_loop("aaa"))  # 6

    print("==== Solution: expanding window")
    print(PalindromicSubstrings().count_palindromic_substr_expanding_window("a"))  # 1
    print(PalindromicSubstrings().count_palindromic_substr_expanding_window("aaa"))  # 6
