"""https://leetcode.com/problems/k-inverse-pairs-array/description/

629. K Inverse Pairs Array
#Hard

Explanation & solution
https://youtu.be/dglwb30bUKI?si=XP4qV4PnttrVuITJ

For an integer array nums, an inverse pair is a pair of integers [i, j]
where 0 <= i < j < nums.length and nums[i] > nums[j].

Given two integers n and k, return the number of different arrays consist of
numbers from 1 to n such that there are exactly k inverse pairs.
Since the answer can be huge, return it modulo 109 + 7.



Example 1:

Input: n = 3, k = 0
Output: 1
Explanation: Only the array [1,2,3] which consists of numbers from 1 to 3 has exactly 0 inverse pairs.
Example 2:

Input: n = 3, k = 1
Output: 2
Explanation: The array [1,3,2] and [2,1,3] have exactly 1 inverse pair.


Constraints:

1 <= n <= 1000
0 <= k <= 1000

"""

from typing import Dict

MOD = 10**9 + 7


class KInversePairsArray:
    def solve_memo(self, n: int, k: int, cache: Dict) -> int:
        """O(n^2 * k) time complexity :("""
        if (n, k) in cache:
            return cache[(n, k)]
        if n == 0:
            return 1 if k == 0 else 0
        if k < 0:
            return 0

        sub_total = 0
        for num_pairs_created in range(n):
            sub_total = (sub_total + self.solve_memo(n - 1, k - num_pairs_created, cache)) % MOD
        cache[(n, k)] = sub_total
        return sub_total

    def solve_dp_full_grid(self, n: int, k: int) -> int:
        """Top-down DP solution

        Base is dp[0][0] = 1

        dp[i][j]: keeps the number of arrays with **at most** k inverse pairs
        for each array of size i and j inverse pairs.
        """
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 1

        for row in range(1, n + 1):
            for col in range(k + 1):
                for pairs_created in range(row):
                    if col - pairs_created >= 0:
                        dp[row][col] += dp[row - 1][col - pairs_created] % MOD
        return dp[n][k]

    def solve_dp_short_grid(self, n: int, k: int) -> int:
        """Top-down DP solution

        To calculate the current line of the grid we only need the previous line.
        Thus, there is no need to store the full dp grid.
        dp is now renamed to previous and is a 1d array of size k + 1

        Base is previous[0] = 1
        """
        previous = [0] * (k + 1)
        previous[0] = 1

        for row in range(1, n + 1):
            current = [0] * (k + 1)
            for col in range(k + 1):
                for pairs_created in range(row):
                    if col - pairs_created >= 0:
                        current[col] += previous[col - pairs_created] % MOD
            previous = current
        return previous[k]

    def solve_dp_short_sliding_window(self, n: int, k: int) -> int:
        """Top-down DP solution
        Building up on top of the previous solution with one line grid,
        we don't need to iterate over all columns, but rather keep a window
        of size n.

        To calculate current[col], we use the previous "line" to look at specific elements
        given a window of scope:
            look up: previous[col]
            look up left: previous[col - 1]
            look up left left: previous[col - 2]
            until a valid window size

        """
        previous = [0] * (k + 1)
        previous[0] = 1

        for row in range(1, n + 1):
            current = [0] * (k + 1)
            window_total = 0
            window_size = row  # we use window_size variable for clarity. We could use row as well
            for col in range(k + 1):
                if col >= window_size:  # here is the trick with the window!
                    window_total -= previous[col - window_size]  # action: remove elements out of window
                window_total = (window_total + previous[col]) % MOD  # action: look up <+ left>
                current[col] = window_total
            previous = current
        return previous[k]
