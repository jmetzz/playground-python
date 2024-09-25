"""https://leetcode.com/problems/n-th-tribonacci-number/description

1137. N-th Tribonacci Number
Easy

The Tribonacci sequence Tn is defined as follows:

T0 = 0, T1 = 1, T2 = 1, and Tn+3 = Tn + Tn+1 + Tn+2 for n >= 0.

Given n, return the value of Tn.

Example 1:
Input: n = 4
Output: 4
Explanation:
T_3 = 0 + 1 + 1 = 2
T_4 = 1 + 1 + 2 = 4

Example 2:
Input: n = 25
Output: 1389537

Constraints:
0 <= n <= 37
The answer is guaranteed to fit within a 32-bit integer, ie. answer <= 2^31 - 1.
"""


def tribonacci_recursive_memoization(n: int) -> int:
    def dfs(target: int, cache: dict) -> int:
        if target == 0:
            return 0
        if target < 3:
            return 1
        if target in cache:
            return cache[target]

        value = dfs(target - 3, cache) + dfs(target - 2, cache) + dfs(target - 1, cache)
        cache[n] = value
        return value

    return dfs(n, {})


def tribonacci_dp_space_order_1_vars(n: int) -> int:
    if n == 0:
        return 0
    if n == 1 or n == 2:
        return 1

    t_zero = 0
    t_one, t_two = 1, 1
    for _ in range(3, n + 1):
        t_i = t_zero + t_one + t_two
        t_zero, t_one, t_two = t_one, t_two, t_i
    return t_i


def tribonacci_dp_space_order_1_list(n: int) -> int:
    previous_values = [0, 1, 1]
    if n <= 2:
        return previous_values[n]
    for _ in range(3, n + 1):
        curr_value = sum(previous_values)
        previous_values[0] = previous_values[1]
        previous_values[1] = previous_values[2]
        previous_values[2] = curr_value
    return curr_value


def tribonacci_dp_space_order_n(n: int) -> int:
    if n == 0:
        return 0
    if n == 1 or n == 2:
        return 1
    dp = [-1 for _ in range(n + 1)]
    dp[0] = 0
    dp[1] = 1
    dp[2] = 1
    for i in range(3, n + 1):
        dp[i] = sum(dp[i - 3 : i])
    return dp[n]


if __name__ == "__main__":
    print(tribonacci_dp_space_order_1_vars(4))
    print(tribonacci_dp_space_order_1_list(4))
    print(tribonacci_dp_space_order_n(4))
    print(tribonacci_recursive_memoization(4))

    print(tribonacci_dp_space_order_1_vars(25))
    print(tribonacci_dp_space_order_1_list(25))
    print(tribonacci_dp_space_order_n(25))
    print(tribonacci_recursive_memoization(25))
