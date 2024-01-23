"""
https://leetcode.com/problems/climbing-stairs/

You are climbing a staircase. It takes n steps to reach the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?



Example 1:

Input: n = 2
Output: 2
Explanation: There are two ways to climb to the top.
1. 1 step + 1 step
2. 2 steps
Example 2:

Input: n = 3
Output: 3
Explanation: There are three ways to climb to the top.
1. 1 step + 1 step + 1 step
2. 1 step + 2 steps
3. 2 steps + 1 step


Constraints:

1 <= n <= 45
"""

def climb_stairs_memoization(n: int) -> int:
    # bottom up using a dp temp storage
    dp = [-1] * (n + 1)
    return solve_with_memoization(0, n, dp)


def solve_with_memoization(i: int, n: int, dp: list) -> int:
    # The base cases
    if i == n:
        return 1
    if i > n:
        return 0
    if dp[i] != -1:
        return dp[i]
    dp[i] = solve_with_memoization(i + 1, n, dp) + solve_with_memoization(i + 2, n, dp)
    return dp[i]


def climb_stairs_tabulation(n: int) -> int:
    # bottom up using a dp temp storage
    # The base cases
    if n == 0 or n == 1:
        return 1
    dp = [-1] * (n + 1)  # sentinel
    dp[0] = dp[1] = 1  # base cases

    for step in range(2, n + 1):
        # current step is the sum of the previous two
        dp[step] = dp[step - 1] + dp[step - 2]
    return dp[n]


def climb_stairs_space_optimization(n: int) -> int:
    # since the calculation at each step only needs
    # the information from the previous two,
    # we don't need to store the complete DP table,
    # as in the solution climb_stairs_tabulation.

    # initializes prev and curr to 1
    # since there is only one way to reach the base cases (0 and 1 steps).
    current = previous = 1

    for step in range(2, n + 1):
        # updates prev and curr by shifting their values. 
        # curr becomes the sum of the previous two values,
        # and prev stores the previous value of curr.
        temp = current + previous
        previous = current
        current = temp

    return current


