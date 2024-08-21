"""https://leetcode.com/problems/longest-ideal-subsequence/description

2370. Longest Ideal Subsequence
Medium

You are given a string s consisting of lowercase letters and an integer k.
We call a string t ideal if the following conditions are satisfied:

t is a subsequence of the string s.
The absolute difference in the alphabet order of every two adjacent letters
in t is less than or equal to k. Return the length of the longest ideal string.

A subsequence is a string that can be derived from another string by deleting
some or no characters without changing the order of the remaining characters.

Note that the alphabet order is not cyclic. For example, the absolute difference
in the alphabet order of 'a' and 'z' is 25, not 1.



Example 1:
Input: s = "acfgbd", k = 2
Output: 4
Explanation: The longest ideal string is "acbd". The length of this string is 4, so 4 is returned.
Note that "acfgbd" is not ideal because 'c' and 'f' have a difference of 3 in alphabet order.

Example 2:
Input: s = "abcd", k = 3
Output: 4
Explanation: The longest ideal string is "abcd". The length of this string is 4, so 4 is returned.


Constraints:

1 <= s.length <= 105
0 <= k <= 25
s consists of lowercase English letters.
"""


def longest_ideal_string_dp_recursive(s: str, k: int) -> int:
    def _dfs(idx, new_letter):
        if dp[idx][new_letter] != -1:
            # return the memoized value
            return dp[idx][new_letter]

        # visit this state (idx, letter)
        match = new_letter == (ord(s[idx]) - 97)
        dp[idx][new_letter] = 1 if match else 0

        if idx > 0:
            dp[idx][new_letter] = _dfs(idx - 1, new_letter)
            if match:
                for p in range(26):
                    if abs(new_letter - p) <= k:
                        dp[idx][new_letter] = max(dp[idx][new_letter], 1 + _dfs(idx - 1, p))
        return dp[idx][new_letter]

    n = len(s)
    # table with n rows and 26 columns to accommodate all possible letters
    dp = [[-1] * 26 for _ in range(n)]  # -1 indicates non-visited states
    answer = 0
    for letter in range(26):
        answer = max(answer, _dfs(n - 1, letter))
    return answer


def longest_ideal_string_dp_space_optimized(s: str, k: int) -> int:
    n = len(s)
    dp = [0] * 26

    answer = 0
    for i in range(n):
        curr_idx = ord(s[i]) - 97
        best = 0
        for prev in range(26):
            if abs(prev - curr_idx) <= k:
                best = max(best, dp[prev])

        dp[curr_idx] = max(dp[curr_idx], best + 1)
        answer = max(answer, dp[curr_idx])
    return answer


def longest_ideal_string_window(s: str, k: int) -> int:
    dp = [0] * 26  # store the maximum length of the ideal subsequence that ends with each character.
    for letter in s:
        idx = ord(letter) - 97
        dp[idx] = 1 + max(dp[max(0, idx - k) : min(26, idx + k + 1)])
    return max(dp)


if __name__ == "__main__":
    print(longest_ideal_string_dp_recursive("acfgbd", 2))
    print(longest_ideal_string_dp_space_optimized("acfgbd", 2))
    print(longest_ideal_string_window("acfgbd", 2))
