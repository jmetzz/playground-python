"""
https://leetcode.com/problems/combinations/description/

77. Combinations
Medium

Given two integers n and k, return all possible combinations of k numbers
chosen from the range [1, n].

You may return the answer in any order.

Example 1:
Input: n = 4, k = 2
Output: [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
Explanation: There are 4 choose 2 = 6 total combinations.
Note that combinations are unordered, i.e., [1,2] and [2,1] are considered
to be the same combination.


Example 2:
Input: n = 1, k = 1
Output: [[1]]
Explanation: There is 1 choose 1 = 1 total combination.


Constraints:

1 <= n <= 20
1 <= k <= n
"""

from itertools import combinations
from typing import List


def combine(n: int, k: int) -> List[List[int]]:
    answer = []

    def backtrack(start: int, comb: List[int]):
        if len(comb) == k:
            answer.append(comb)
            return

        for i in range(start + 1, n + 1):
            backtrack(i, comb + [i])

    backtrack(0, [])
    return answer


def combine_itertools(n: int, k: int) -> List[List[int]]:
    return list(combinations(range(1, n + 1), k))


if __name__ == "__main__":
    print(combine(4, 2))  # [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
    print(combine(1, 1))  # [[1]]
