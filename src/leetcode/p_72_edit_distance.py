"""https://leetcode.com/problems/edit-distance/description

72. Edit Distance
Medium

Given two strings word1 and word2, return the minimum number of
operations required to convert word1 to word2.

You have the following three operations permitted on a word:

Insert a character
Delete a character
Replace a character


Example 1:
Input: word1 = "horse", word2 = "ros"
Output: 3
Explanation:
horse -> rorse (replace 'h' with 'r')
rorse -> rose (remove 'r')
rose -> ros (remove 'e')

Example 2:
Input: word1 = "intention", word2 = "execution"
Output: 5
Explanation:
intention -> inention (remove 't')
inention -> enention (replace 'i' with 'e')
enention -> exention (replace 'n' with 'x')
exention -> exection (replace 'n' with 'c')
exection -> execution (insert 'u')


Constraints:
0 <= word1.length, word2.length <= 500
word1 and word2 consist of lowercase English letters.
"""

from functools import cache
from math import inf


def edit_distance(word1: str, word2: str) -> int:
    """Given text1 = abcde and text2 = ace
    form a dp cache table as:

        dp = [    a  c  e
             a   [0, 0, 0 | 3]
             b   [0, 0, 0 | 2]
             d   [0, 0, 0 | 1]
                 ---------|--
                 [3, 2, 1 | 0]
            ]

    Then iterate updating the values according to:
    a) first chars match
    b) first chars do not match

        if w1[i] == w2[j]:
            i++
            j++
        else:
            insert: op_count++ & j++
            delete: op_count++ & i++
            replace: op_count++ & i++ & j++

    """
    n, m = len(word1), len(word2)
    dp = [[inf] * (m + 1) for _ in range(n + 1)]
    # fill the bottom row
    dp[-1] = [i for i in reversed(range(m + 1))]
    # fill in the right most column
    for i in range(n):
        dp[i][-1] = n - i

    # bottom up dynamic programming block
    for i in reversed(range(n)):
        for j in reversed(range(m)):
            if word1[i] == word2[j]:
                # match -> no operation
                dp[i][j] = dp[i + 1][j + 1]
            else:
                # no match -> 1 + the min of other operations
                dp[i][j] = 1 + min(
                    dp[i + 1][j],  # delete
                    dp[i][j + 1],  # insert
                    dp[i + 1][j + 1],  # replace
                )
    return dp[0][0]


def edit_distance_recursive(word1: str, word2: str) -> int:
    m, n = len(word1), len(word2)

    @cache
    def helper(i, j):
        if i == 0:
            # word1 is "" --> insert for all j characters, thus, j operations
            return j
        if j == 0:
            # word2 is "" --> delete all i characters to match "", thus, i operations
            return i
        if word1[i - 1] == word2[j - 1]:
            # match case -> process sub-problems & no operations needed
            return helper(i - 1, j - 1)
        return 1 + min(
            helper(i, j - 1),  # insert
            helper(i - 1, j),  # delete
            helper(i - 1, j - 1),  # replace
        )

    return helper(m, n)


if __name__ == "__main__":
    print(edit_distance("abd", "acd"))
    print(edit_distance_recursive("abd", "acd"))
