import heapq
import operator
from collections import Counter
from typing import List


def least_interval_explained(tasks: List[str], n: int) -> int:
    # ["A","A","A","B","B","B"], n = 2
    counter = Counter(tasks)
    sorted_tasks = sorted(counter.items(), key=operator.itemgetter(1), reverse=True)
    # sorted_tasks = [(A, 3), (B, 3)]
    heap = []
    for task, qtt in sorted_tasks:
        for i in range(qtt):
            heap.append((i * (n + 1), task))
    # heap = [(0, A), (3, A), (6, A), (0, B), (3, B), (6, B)]
    heapq.heapify(heap)
    # heap = [(0, A), (0, B), (3, A), (3, B), (6, A), (6, B)]
    answer = []
    i = 0
    while heap:
        if i >= heap[0][0]:
            answer.append(heapq.heappop(heap))
        else:
            answer.append("idle")
        i += 1

    # i=0 --> answer[A]
    # i=1 --> answer[A, B]
    # i=2 --> answer[A, B, idle]
    # i=3 --> answer[A, B, idle, A]
    # i=4 --> answer[A, B, idle, A, B]
    # i=5 --> answer[A, B, idle, A, B, idle]
    # i=6 --> answer[A, B, idle, A, B, idle, A]
    # i=7 --> answer[A, B, idle, A, B, idle, A, B]
    return answer


def least_interval_max_heap(tasks: List[str], n: int) -> int:
    counter = Counter(tasks)
    sorted_tasks = sorted(counter.items(), key=operator.itemgetter(1), reverse=True)
    heap = []
    for task, qtt in sorted_tasks:
        for intervals in range(qtt):
            heap.append((intervals * (n + 1), task))
    heapq.heapify(heap)
    intervals = 0
    while heap:
        if intervals >= heap[0][0]:
            heapq.heappop(heap)
        # else:
        #    --> "idle"
        intervals += 1
    return intervals


def least_interval_max_heap_2(tasks: List[str], n: int) -> int:
    """Approach optimizes efficiency by prioritizing tasks
    based on their frequency, thereby reducing intervals and
    minimizing idle time for the scheduler.

    Time: O(n)
    Space: O(1)
    """
    counter = Counter(tasks)
    max_heap = [-c for c in counter.values()]
    heapq.heapify(max_heap)
    total_time = 0
    while max_heap:
        cycle = n + 1
        task_count = 0
        temp_arr = []
        while cycle > 0 and max_heap:
            c = -heapq.heappop(max_heap)
            if c > 1:
                temp_arr.append(-(c - 1))
            task_count += 1
            cycle -= 1
        # for new_task_count in temp_arr:
        #     heapq.heappush(max_heap, new_task_count)
        max_heap.extend(temp_arr)
        heapq.heapify(max_heap)

        total_time += task_count if not max_heap else n + 1
    return total_time


def least_interval_chunck_based(tasks: List[str], n: int) -> int:
    """Find the minimum number of intervals required to complete all tasks
    while following the cooling time constraint.

    1. count the frequency of each task.
    2. sort the frequencies in descending order to prioritize tasks with higher frequency.
    3. calculate the number of intervals needed by considering the task with the max frequency.

    The key point for understanding the solution is the "Reducing Idle Time" step
    (the 'for i in range(24, -1, -1):' below), which is also crucial for optimizing the
    task scheduling and minimize the total execution time.

    Here is the intuition:

    > Initial Idle Time Calculation
    Initially, total_idle_needed is calculated based on the most frequent task.
    Imagine you have the most frequent task 'A' that appears 4 times in your task list,
    and your cooldown period (n) is 2. Ignoring other tasks, your schedule might look like
        A _ _ A _ _ A _ _ A,
    where _ represents idle slots.
    Here, the "chunk" size is 3 (the frequency of 'A' minus 1), leading to
        total_idle_needed = chunk_size * n = 3 * 2 = 6
    idle slots.

    > Adjusting for Other Tasks
    Now, consider you have other tasks ('B', 'C', etc.) that can fill these idle slots.
    The key insight is that for each slot you fill, you reduce the total_idle_needed by 1.
    However, once you've filled all idle slots corresponding to the intervals between 'A's,
    additional 'B's or 'C's don't further reduce idle time (but they don't increase it either,
    as they simply extend the total schedule length).
    The algorithm iterates over the frequencies of the remaining tasks (in descending order)
    and tries to "fill" the idle slots created by the most frequent task ('A'in this example).

    If its frequency is less than or equal to chunk (the number of idle slots between consecutive 'A's),
    it can fully occupy some or all of those idle slots, directly reducing
    total_idle_needed by the task's frequency, using at most 1 per chunk.

    After considering all tasks, if total_idle_needed is positive, it means not all idle slots
    could be filled, and this remaining idle time is added to the total.
    If total_idle_needed is zero or negative, all tasks could be arranged within the minimum
    required intervals, so the total time equals the number of tasks.

    Time: O(26 log 26) --> O(1)
    Space: O(1)
    """
    if not tasks:
        return 0
    freq = [0] * 26  # for all tasks are in [A, Z]
    for task in tasks:
        freq[ord(task) - ord("A")] += 1
    freq.sort()
    # The task with the maximum frequency (freq[25] after sorting)
    # defines the initial structure of the scheduling.
    # If the most frequent task occurs 'f' times, you need
    # at least 'f-1' chunks of time, each of size n (the cooldown period),
    # plus 1 more interval for the last execution of this task.
    chunk_size = freq[-1] - 1
    total_idle_needed = chunk_size * n

    for i in range(24, -1, -1):
        # adapts the initial idle estimate based on less frequent tasks
        total_idle_needed -= min(chunk_size, freq[i])
        # If total_idle_needed is less than or equal to 0, all tasks could
        # be scheduled back to back without any idle time

    # before returning, check for scenarios
    # where all tasks can be scheduled without remaining idle slots
    return len(tasks) + total_idle_needed if total_idle_needed >= 0 else len(tasks)


print(least_interval_explained(["A", "A", "A", "B", "B", "B"], n=2))
print(least_interval_max_heap(["A", "A", "A", "B", "B", "B"], 2))
# --
print(least_interval_max_heap_2(["A", "C", "A", "B", "D", "B"], n=1))
print(least_interval_max_heap_2(["A", "A", "A", "B", "B", "B"], n=3))
