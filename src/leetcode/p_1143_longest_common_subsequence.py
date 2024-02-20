"""
https://www.youtube.com/watch?v=Ua0GhsJSlWM

https://leetcode.com/problems/longest-common-subsequence/description/

1143. Longest Common Subsequence

Given two strings text1 and text2, return the length of their longest common subsequence.
If there is no common subsequence, return 0.

A subsequence of a string is a new string generated from the original string
with some characters (can be none) deleted without changing the relative
order of the remaining characters.

For example, "ace" is a subsequence of "abcde".
A common subsequence of two strings is a subsequence that is common to both strings.



Example 1:

Input: text1 = "abcde", text2 = "ace"
Output: 3
Explanation: The longest common subsequence is "ace" and its length is 3.
Example 2:

Input: text1 = "abc", text2 = "abc"
Output: 3
Explanation: The longest common subsequence is "abc" and its length is 3.
Example 3:

Input: text1 = "abc", text2 = "def"
Output: 0
Explanation: There is no such common subsequence, so the result is 0.


Constraints:

1 <= text1.length, text2.length <= 1000
text1 and text2 consist of only lowercase English characters.
"""


class LongestCommonSubsequence:
    def solve(self, text1: str, text2: str) -> int:
        """

        O(n*m) time
        O(n*m) space

        given text1 = abcde and text2 = ace
        dp = [    a  c  e
             a   [0, 0, 0 | 0]
             b   [0, 0, 0 | 0]
             c   [0, 0, 0 | 0]
             d   [0, 0, 0 | 0]
             e   [0, 0, 0 | 0]
                 ---------|--
                 [0, 0, 0 | 0]
            ]

        dp has 1 extra column and 1 extra line.

        """
        n = len(text1)
        m = len(text2)
        dp = [[0 for _ in range(m + 1)] for _ in range(n + 1)]

        # Using a bottom up approach, we start the right lower corner of the matrix,
        # (not counting the extra column and row) which corresponds to
        # the last character in each input sequences, and iterate upwards to the
        # first position [0][0], which will result in the final solution.
        for i in range(n - 1, -1, -1):
            for j in range(m - 1, -1, -1):
                if text1[i] == text2[j]:  # move diagonal
                    dp[i][j] = 1 + dp[i + 1][j + 1]
                else:
                    # checking down and right solutions, keeping the max value
                    dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])

        return dp[0][0]


if __name__ == "__main__":
    print(LongestCommonSubsequence().solve("abcde", "ace"))
