from typing import List


class Solution:
    def solve_memoization(self, nums: List[int]) -> List[int]:
        """
        Model the problem as "take or not" the element at index i.

        answer[i] % answer[j] == 0 OR answer[j] % answer[i] == 0
        means that one number is a factor of the other.

        If we sort the numbers and assume i <= j,
        the condition becomes answer[i] % answer[j] == 0.
        """
        cache = {}  # (idx, previous) -> List
        nums.sort()  # sort in place to save space. Side effect is bad!

        def dfs(idx, previous) -> List[int]:
            if idx == len(nums):
                return []

            if (idx, previous) in cache:
                return cache[(idx, previous)]

            res = dfs(idx + 1, previous)  # don't take i

            if nums[idx] % previous == 0:
                # take the element at index i and add to the solution
                # nums[i] becomes previous in the next branch (recursive call)
                tmp = [nums[idx]] + dfs(idx + 1, nums[idx])
                if len(tmp) > len(res):
                    res = tmp

            cache[(idx, previous)] = res
            return res

        return dfs(0, 1)

    def solve_memoization_optimized(self, nums: List[int]) -> List[int]:

        nums.sort()  # sort in place to save space. Side effect is bad!
        n = len(nums)

        # cache[i] = longest subset starting at i, and including nums[i]
        cache = {}

        def dfs(idx) -> List[int]:
            if idx == n:
                return []

            if idx in cache:
                return cache[idx]

            res = [nums[idx]]
            for j in range(idx + 1, n):
                if nums[j] % nums[idx] == 0:
                    tmp = [nums[idx]] + dfs(j)  # take nums[i]
                    if len(tmp) > len(res):
                        res = tmp
            cache[idx] = res
            return res

        answer = []
        for i in range(n):
            tmp = dfs(i)
            if len(tmp) > len(answer):
                answer = tmp

        return answer

    def solve_dp_bottom_up(self, nums: List[int]) -> List[int]:
        nums.sort()  # sort in place to save space. Side effect is bad!
        n = len(nums)
        dp = [[e] for e in nums]
        longest = []
        for i in range(n - 1, -1, -1):  # reverse loop
            # for i in reversed(range(n)):  # reverse loop
            for j in range(i + 1, n):
                if nums[j] % nums[i] == 0:
                    temp = [nums[i]] + dp[j]
                    if len(temp) > len(dp[i]):
                        dp[i] = temp
            if len(dp[i]) > len(longest):
                longest = dp[i]
        return longest


if __name__ == '__main__':
    print(Solution().solve_memoization([1, 2, 3]))
    print(Solution().solve_memoization_optimized([1, 2, 3]))
    print(Solution().solve_dp_bottom_up([1, 2, 3]))
