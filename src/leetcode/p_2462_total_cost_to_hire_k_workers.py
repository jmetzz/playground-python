"""
https://leetcode.com/problems/total-cost-to-hire-k-workers/description
2462. Total Cost to Hire K Workers
Medium

You are given a 0-indexed integer array costs where costs[i] is the cost of
hiring the ith worker.

You are also given two integers k and candidates. We want to hire exactly
k workers according to the following rules:

You will run k sessions and hire exactly one worker in each session.
In each hiring session, choose the worker with the lowest cost from either
the first candidates workers or the last candidates workers.
Break the tie by the smallest index.

For example, if costs = [3,2,7,7,1,2] and candidates = 2, then in the first hiring session,
we will choose the 4th worker because they have the lowest cost [3,2,7,7,1,2].
In the second hiring session, we will choose 1st worker because they have the
same lowest cost as 4th worker but they have the smallest index [3,2,7,7,2].
Please note that the indexing may be changed in the process.
If there are fewer than candidates workers remaining, choose the worker with
the lowest cost among them. Break the tie by the smallest index.
A worker can only be chosen once.
Return the total cost to hire exactly k workers.



Example 1:

Input: costs = [17,12,10,2,7,2,11,20,8], k = 3, candidates = 4
Output: 11
Explanation: We hire 3 workers in total. The total cost is initially 0.
- In the first hiring round we choose the worker from [17,12,10,2,7,2,11,20,8].
The lowest cost is 2, and we break the tie by the smallest index, which is 3.
The total cost = 0 + 2 = 2.
- In the second hiring round we choose the worker from [17,12,10,7,2,11,20,8].
The lowest cost is 2 (index 4). The total cost = 2 + 2 = 4.
- In the third hiring round we choose the worker from [17,12,10,7,11,20,8].
The lowest cost is 7 (index 3). The total cost = 4 + 7 = 11.
Notice that the worker with index 3 was common in the first and last four workers.
The total hiring cost is 11.
Example 2:

Input: costs = [1,2,4,1], k = 3, candidates = 3
Output: 4
Explanation: We hire 3 workers in total. The total cost is initially 0.
- In the first hiring round we choose the worker from [1,2,4,1].
The lowest cost is 1, and we break the tie by the smallest index, which is 0.
The total cost = 0 + 1 = 1. Notice that workers with index 1 and 2 are
common in the first and last 3 workers.
- In the second hiring round we choose the worker from [2,4,1].
The lowest cost is 1 (index 2). The total cost = 1 + 1 = 2.
- In the third hiring round there are less than three candidates.
We choose the worker from the remaining workers [2,4].
The lowest cost is 2 (index 0). The total cost = 2 + 2 = 4.
The total hiring cost is 4.


Constraints:

1 <= costs.length <= 105
1 <= costs[i] <= 105
1 <= k, candidates <= costs.length
"""

from heapq import heapify, heappop, heappush
from typing import List


def total_cost_one_heap(costs: List[int], k: int, candidates: int) -> int:
    n = len(costs)
    left, right = 0, n

    # build the session min heap. Each element is the (cost, original index)
    if candidates * 2 >= n:
        session = [(c, i) for i, c in enumerate(costs)]
        left, right = right, left  # cross the pointer to symbolise there are no more elements to be added
    else:
        session = []
        for left in range(candidates):
            session.append((costs[left], left))
        for right in range(n - 1, n - candidates - 1, -1):
            session.append((costs[right], right))
        # equivalent to:
        # tail_start = max(candidates, len(costs) - candidates)
        # session = ( list(zip(costs[:candidates], range(candidates)))
        #           + list(zip(costs[tail_start  :], range(tail_start, n))))

    # make session a min heap
    heapify(session)

    hires = 0
    hiring_cost = 0
    while hires < k and session:
        cost, idx = heappop(session)
        hiring_cost += cost
        hires += 1

        if left < right - 1:
            if idx <= left:
                # consumed element from left side
                left += 1
                heappush(session, (costs[left], left))
            else:
                # consumed element from right side
                right -= 1
                heappush(session, (costs[right], right))

    return hiring_cost


def total_cost_two_heaps(costs: List[int], k: int, candidates: int) -> int:
    head_workers = costs[:candidates]
    tail_workers = costs[max(candidates, len(costs) - candidates) :]
    heapify(head_workers)
    heapify(tail_workers)

    hiring_cost = 0
    left, right = candidates, len(costs) - 1 - candidates

    for _ in range(k):
        # short circuit for efficiency
        if not tail_workers or head_workers and head_workers[0] <= tail_workers[0]:
            hiring_cost += heappop(head_workers)

            if left <= right:
                # consumed element from left side (head)
                heappush(head_workers, costs[left])
                left += 1
        else:
            hiring_cost += heappop(tail_workers)
            #  Only refill the queue if there are workers outside the two queues.
            if left <= right:
                # consumed element from right side (tail)
                heappush(tail_workers, costs[right])
                right -= 1
    return hiring_cost


# print(total_cost([17, 12, 10, 2, 7, 2, 11, 20, 8], k=3, candidates=4))  # 11
# print(total_cost([1, 2, 4, 1], k=3, candidates=3))  # 4
