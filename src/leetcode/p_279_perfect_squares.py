import math
from collections import deque


class Solution:
    def numSquares(self, n: int) -> int:
        psquare = [i * i for i in range(1, int(math.sqrt(n)) + 1)]
        fringe = deque([[v] for v in psquare])
        while fringe:
            candidate = fringe.popleft()
            curr_sum = sum(candidate)
            if curr_sum == n:
                return len(candidate)
            fringe.extend([candidate + [v] for v in psquare if curr_sum + v <= n])
        return 0

    def numSquares_dp(self, n: int) -> int:
        dp = [float("inf")] * (n + 1)
        dp[0] = 0

        for i in range(1, n + 1):
            min_value = float("inf")
            j = 1
            while j * j <= i:
                perf_sq = j * j
                min_value = min(min_value, dp[i - perf_sq] + 1)
                j += 1
            dp[i] = min_value
        return dp[n]


if __name__ == "__main__":
    print(Solution().numSquares(12))
    print(Solution().numSquares_dp(12))
    print(Solution().numSquares_dp(7168))
