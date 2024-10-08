"""https://leetcode.com/problems/time-needed-to-buy-tickets

2073. Time Needed to Buy Tickets
Easy

There are n people in a line queuing to buy tickets, where the 0th person is
at the front of the line and the (n - 1)th person is at the back of the line.

You are given a 0-indexed integer array tickets of length n where the
number of tickets that the ith person would like to buy is tickets[i].

Each person takes exactly 1 second to buy a ticket. A person can only
buy 1 ticket at a time and has to go back to the end of the line
(which happens instantaneously) in order to buy more tickets.
If a person does not have any tickets left to buy, the person will leave the line.

Return the time taken for the person at position k (0-indexed) to
finish buying tickets.



Example 1:

Input: tickets = [2,3,2], k = 2
Output: 6
Explanation:
- In the first pass, everyone in the line buys a ticket and the line becomes [1, 2, 1].
- In the second pass, everyone in the line buys a ticket and the line becomes [0, 1, 0].
The person at position 2 has successfully bought 2 tickets and it took 3 + 3 = 6 seconds.
Example 2:

Input: tickets = [5,1,1,1], k = 0
Output: 8
Explanation:
- In the first pass, everyone in the line buys a ticket and the line becomes [4, 0, 0, 0].
- In the next 4 passes, only the person in position 0 is buying tickets.
The person at position 0 has successfully bought 5 tickets and it took 4 + 1 + 1 + 1 + 1 = 8 seconds.


Constraints:

n == tickets.length
1 <= n <= 100
1 <= tickets[i] <= 100
0 <= k < n
"""

from collections import deque


def time_required_to_buy_1(tickets: list[int], k: int) -> int:
    if tickets[k] == 1:
        return k + 1

    n = len(tickets)
    index_queue = deque(range(n))

    total_time = 0
    front = 0
    while index_queue:
        total_time += 1
        front = index_queue.popleft()
        tickets[front] -= 1
        # stop as soon as k has got all the desired tickets
        if front == k and tickets[k] == 0:
            return total_time
        if tickets[front] != 0:
            index_queue.append(front)
    return total_time


def time_required_to_buy_2(tickets: list[int], k: int) -> int:
    if tickets[k] == 1:
        return k + 1

    n = len(tickets)
    total_time = 0

    while tickets[k] > 0:
        for i in range(n):
            if tickets[i] != 0:
                tickets[i] -= 1
                total_time += 1
            if tickets[k] == 0:
                return total_time
    return total_time


def time_required_to_buy_3(tickets: list[int], k: int) -> int:
    if tickets[k] == 1:
        return k + 1

    total_time = 0

    # first case: current person is before k
    for i in range(k + 1):
        total_time += min(tickets[i], tickets[k])

    # second case: current person is after k
    for i in range(k + 1, len(tickets)):
        # If the current person is after 'k', buy the minimum of
        # (tickets available at person 'k' - 1) and the current person
        total_time += min(tickets[k] - 1, tickets[i])

    return total_time
