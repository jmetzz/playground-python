"""
1043. Partition Array for Maximum Sum

Given an integer array arr, partition the array into (contiguous)
subarrays of length at most k. After partitioning,
each subarray has their values changed to become
the maximum value of that subarray.


Return the largest sum of the given array after partitioning.
Test cases are generated so that the answer fits in a 32-bit integer.

Example 1:
Input: arr = [1,15,7,9,2,5,10], k = 3
Output: 84
Explanation: arr becomes [15,15,15,9,10,10,10]

Example 2:
Input: arr = [1,4,1,5,7,3,6,1,9,9,3], k = 4
Output: 83

Example 3:
Input: arr = [1], k = 1
Output: 1


Constraints:
1 <= arr.length <= 500
0 <= arr[i] <= 109
1 <= k <= arr.length


Explanation
-----------

Overview
We have an array of N integers which we can partition into any number of subarrays
such that each subarray can have at most a length of k. After partitioning the array,
the values in each subarray will change to the maximum value in that subarray.
We need to find the maximum sum of all these subarrays.

We can observe that for each element, we have two options: choose this element in
the current subarray or choose to end the current subarray before this element and
start another one from this element. The brute force approach is to enumerate
every possible combination.

There are two key characteristics of this problem that we should note.
First, if we choose an element in a subarray, it cannot be reused in another subarray,
i.e., each decision we make is affected by the previous decisions we have made.
Second, the problem asks us to maximize the sum when choosing the subarrays.
These are two common characteristics of dynamic programming problems,
and as such we will approach this problem using dynamic programming.

Dynamic programming is a programming paradigm in which we break a problem into sub-problems,
store the result of each sub-problem, and use it when required.


Approach: Top-Down Dynamic Programming

At every index, we can decide the length of the subarray with this index as the starting point,
denoted as start. We can choose a subarray of any length from 1 to k.

Let's start from the index 0; we will iterate over the elements and keep the
maximum value we have found so far in the variable currMax. When we choose to end the subarray,
we will assume each element in it to be the maximum value in that subarray.
For each element, we will find the total sum if we choose to keep this subarray.
This will be equal to the sum of the current subarray and the maximum sum we can
get from the rest of the array.

The sum of the current subarray will be currMax * length of subarray because
each element's value will be changed to currMax. For the sum of the remaining array,
we will make the recursive call to the function with the next index as the
starting element of the array. For each index, we will choose subarrays of all lengths
up to k and return the maximum of all these options. The base condition in the
recursive function would be when we have iterated over the complete array
in which case we should return 0.

This recursive approach will have repeated subproblems (subtrees), signifying
that we must solve these subproblems more than once.

Each node in the image represents an index of arr.

To address this issue, the first time we calculate sum for a certain index,
we will store the value in an array; this value represents the maximum sum
we can get from the elements at indices from the start index to the end of the array.
The next time we need to calculate the sum for this position,
we can look up the result in constant time. This technique is known as memoization,
and it helps us avoid recalculating repeated subproblems.


Complexity Analysis
Let N be the number of elements in the array, and K is the maximum length of a subarray.

Time complexity: O(N⋅K)

The time complexity for the recursion with memoization is equal to
the number of times maxSum() is called times the average time of maxSum().
The number of calls to maxSum() is N because each non-memoized call
will call maxSum() only once. Each memoized call will take O(1) time,
while for the non-memoized call, we iterate over most KKK elements ahead of it.
Hence, the total time complexity equals O(N⋅K)

Space complexity: O(N)

The result for each start index will be stored in dp,
and start can have the values from 0 to N; thus,
the space required is O(N).
Also, the stack space needed for recursion is equal to the
maximum number of active function calls which will be N,
one for each index. Hence, the space complexity will equal O(N).
"""
from typing import List


class PartitionArrayForMaxSum:
    def solve_dfs(self, arr: List[int], k: int) -> int:
        def _dfs(start_idx: int) -> int:
            if start_idx >= len(arr):
                # base case -- which is made redundant because of the
                # min(n, i + k) in the loop control
                return 0

            cur_max = 0
            answer = 0
            end_idx = min(len(arr), start_idx + k)
            for j in range(start_idx, end_idx):
                cur_max = max(cur_max, arr[j])
                win_size = j - start_idx + 1
                answer = max(answer, _dfs(j + 1) + cur_max * win_size)
            return answer

        return _dfs(0)

    def solve_dfs_dict_memo(self, arr: List[int], k: int) -> int:
        cache = {}

        def _dfs(start_idx: int) -> int:
            if start_idx >= len(arr):
                # base case -- which is made redundant because of the
                # min(n, i + k) in the loop control
                return 0

            if start_idx in cache:
                return cache[start_idx]

            cur_max = 0
            answer = 0
            end_idx = min(len(arr), start_idx + k)
            for j in range(start_idx, end_idx):
                cur_max = max(cur_max, arr[j])
                win_size = j - start_idx + 1
                answer = max(answer, _dfs(j + 1) + cur_max * win_size)
            cache[start_idx] = answer
            return answer

        return _dfs(0)

    def solve_dfs_list_memo(self, arr: List[int], k: int) -> int:
        table = [-1] * len(arr)  # -1 indicates the answer is not calculated yet

        def _dfs(start_idx: int) -> int:
            if start_idx >= len(arr):
                # base case -- which is made redundant because of the
                # min(n, i + k) in the loop control
                return 0

            if table[start_idx] != -1:
                return table[start_idx]

            curr_max = 0
            answer = 0
            end_idx = min(start_idx + k, len(arr))
            for idx in range(start_idx, end_idx):
                curr_max = max(curr_max, arr[idx])
                answer = max(answer, curr_max * (idx - start_idx + 1) + _dfs(idx + 1))

            table[start_idx] = answer
            return answer

        return _dfs(0)

    def solve_tabulation_top_down(self, arr: List[int], k: int) -> int:
        n = len(arr)
        dp = [0] * n
        dp[0] = 0  # base case
        for i in range(1, n):
            cur_max = 0
            max_value_at_i = 0
            for j in range(i, i - k, -1):
                if j < 0:
                    break
                cur_max = max(cur_max, arr[j])
                win_size = i - j + 1
                cur_sum = cur_max * win_size
                # dp[j - 1] represents the sub-problem
                max_value_at_i = max(max_value_at_i, cur_sum + dp[j - 1])
            dp[i] = max_value_at_i
        return dp[-1]

    def solve_tabulation_top_down_circular_list(self, arr: List[int], k: int) -> int:
        n = len(arr)
        dp = [0] * k
        dp[0] = arr[0]  # base case

        for i in range(1, n):
            # i pointer represents the end of the window
            cur_max = 0
            max_value_at_i = 0
            for j in range(i, i - k, -1):
                # j pointer represents the beginning of the window (looking left approach)
                if j < 0:
                    break
                cur_max = max(cur_max, arr[j])
                window_size = i - j + 1
                cur_sum = cur_max * window_size
                # get the result of the sub-problem
                # sub_sum = dp[j - 1] if j > 0 else 0
                # using a circular array:
                sub_sum = dp[(j - 1) % k] if j > 0 else dp[-1]
                max_value_at_i = max(max_value_at_i, cur_sum + sub_sum)
            dp[i % k] = max_value_at_i

        return dp[(n - 1) % k]
