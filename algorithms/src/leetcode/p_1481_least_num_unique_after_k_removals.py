"""
https://leetcode.com/problems/least-number-of-unique-integers-after-k-removals/description

1481. Least Number of Unique Integers after K Removals
Medium

Given an array of integers arr and an integer k.
Find the least number of unique integers after removing exactly k elements.



Example 1:
Input: arr = [5,5,4], k = 1
Output: 1
Explanation: Remove the single 4, only 5 is left.

Example 2:
Input: arr = [4,3,1,1,3,3,2], k = 3
Output: 2
Explanation: Remove 4, 2 and either one of the two 1s or three 3s. 1 and 3 will be left.


Constraints:
1 <= arr.length <= 10^5
1 <= arr[i] <= 10^9
0 <= k <= arr.length

"""
import heapq
from collections import Counter
from typing import List


class FindLeastNumOfUniqueAfterRemoval:
    def solve_with_heap(self, arr: List[int], k: int) -> int:
        counter = Counter(arr)
        heap = [(count, e) for e, count in counter.items()]
        heapq.heapify(heap)

        while k > 0 and heap:
            count, e = heapq.heappop(heap)
            if count <= k:
                # take all: do not push the element back
                k -= count
            else:
                # take k elements
                heapq.heappush(heap, (count - k, e))
                k = 0
        unique_elements = {e for _, e in heap}
        return len(unique_elements)

    def solve_with_array(self, arr: List[int], k: int) -> int:
        counter = Counter(arr)
        temp = sorted([(count, e) for e, count in counter.items()])
        idx = 0
        while k > 0 and idx < len(temp):
            count, e = temp[idx]
            if count > k:
                # "take" k elements and stop the loop
                break
            # k >= count
            # take all elements of e and "remove it"
            k -= count
            idx += 1
        return len(temp) - idx

    def solve_with_one_loop(self, arr: List[int], k: int) -> int:
        counter = Counter(arr)
        counts = sorted([count for count in counter.values()])
        idx = 0
        while k > 0 and idx < len(counts):
            if counts[idx] > k:
                # "take" k elements and stop the loop
                break
            # otherwise, when k >= count
            # take all elements of e and "remove it"
            k -= counts[idx]
            idx += 1
        return len(counts) - idx

    def solve_with_counter_and_counter(self, arr: List[int], k: int) -> int:
        """
        This one does not work properly yet
        failed for this test case:
        [2,4,1,8,3,5,1,3], k=3 --> expected is 3
        """
        counter = Counter(arr)
        counter = sorted([(count, size) for count, size in Counter([c for c in counter.values()]).items()])
        idx = 0
        residual = 0
        while k > 0 and idx < len(counter):
            count, size = counter[idx]
            n_candidates = count * size
            if n_candidates > k:
                # "take" k elements and stop the loop
                residual = n_candidates - k
                idx += 1
                break
            # otherwise, when k >= count
            # take all elements of e and "remove it"
            k -= n_candidates
            idx += 1
        remaining_elements = len(counter) - idx
        return remaining_elements + residual

    def solve_ffreitas(self, arr: List[int], k: int) -> int:
        k_count = k
        c = Counter(arr)
        cnt, remaining = Counter(c.values()), len(c)

        for key in range(1, len(arr) + 1):
            if k_count >= key * cnt[key]:
                k_count -= key * cnt[key]
                remaining -= cnt[key]
            else:
                return remaining - k_count // key

        return remaining
