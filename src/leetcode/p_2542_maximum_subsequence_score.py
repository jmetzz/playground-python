"""https://leetcode.com/problems/maximum-subsequence-score/description

2542. Maximum Subsequence Score
Medium

Topics: Array, Greedy, Sorting, Heap (Priority Queue)

You are given two 0-indexed integer arrays nums1 and nums2 of equal length n and a positive integer k.
ou must choose a subsequence of indices from nums1 of length k.

For chosen indices i0, i1, ..., ik - 1, your score is defined as:

The sum of the selected elements from nums1 multiplied with the minimum of the selected elements from nums2.
It can defined simply as: (nums1[i0] + nums1[i1] +...+ nums1[ik - 1]) * min(nums2[i0] , nums2[i1], ... ,nums2[ik - 1]).
Return the maximum possible score.

A subsequence of indices of an array is a set that can be derived from the set {0, 1, ..., n-1} by
deleting some or no elements.



Example 1:
Input: nums1 = [1,3,3,2], nums2 = [2,1,3,4], k = 3
Output: 12
Explanation:
The four possible subsequence scores are:
- We choose the indices 0, 1, and 2 with score = (1+3+3) * min(2,1,3) = 7.
- We choose the indices 0, 1, and 3 with score = (1+3+2) * min(2,1,4) = 6.
- We choose the indices 0, 2, and 3 with score = (1+3+2) * min(2,3,4) = 12.
- We choose the indices 1, 2, and 3 with score = (3+3+2) * min(1,3,4) = 8.
Therefore, we return the max score, which is 12.

Example 2:
Input: nums1 = [4,2,3,1,1], nums2 = [7,5,10,9,6], k = 1
Output: 30
Explanation:
Choosing index 2 is optimal: nums1[2] * nums2[2] = 3 * 10 = 30 is the maximum possible score.


Constraints:

n == nums1.length == nums2.length
1 <= n <= 105
0 <= nums1[i], nums2[j] <= 105
1 <= k <= n



Hint 1: How can we use sorting here?
Hint 2: Try sorting the two arrays based on second array.
Hint 3: Loop through nums2 and compute the max product given the minimum is nums2[i]. Update the answer accordingly.

Explanation: https://youtu.be/ax1DKi5lJwk?si=cOFrq9_C8UbztDCC

"""

import operator
from heapq import heappop, heappush


def max_score(nums1: list[int], nums2: list[int], k: int) -> int:
    pairs = sorted(list(zip(nums1, nums2, strict=False)), key=operator.itemgetter(1), reverse=True)

    context_heap = []
    nums1_running_sum, answer = 0, 0

    for n1, n2 in pairs:
        nums1_running_sum += n1
        heappush(context_heap, n1)

        if len(context_heap) > k:
            nums1_running_sum -= heappop(context_heap)

        if len(context_heap) == k:
            answer = max(answer, nums1_running_sum * n2)
    return answer
