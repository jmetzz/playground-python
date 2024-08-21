"""https://leetcode.com/problems/find-all-people-with-secret/description

2092. Find All People With Secret
Hard

You are given an integer n indicating there are n people numbered from 0 to n - 1.
You are also given a 0-indexed 2D integer array meetings where meetings[i] = [xi, yi, timei]
indicates that person xi and person yi have a meeting at timei. A person may attend multiple
meetings at the same time. Finally, you are given an integer firstPerson.

Person 0 has a secret and initially shares the secret with a person firstPerson at time 0.
This secret is then shared every time a meeting takes place with a person that has the secret.
More formally, for every meeting, if a person xi has the secret at timei, then they will
share the secret with person yi, and vice versa.

The secrets are shared instantaneously. That is, a person may receive the secret and share
it with people in other meetings within the same time frame.

Return a list of all the people that have the secret after all the meetings have taken place.
You may return the answer in any order.



Example 1:
Input: n = 6, meetings = [[1,2,5],[2,3,8],[1,5,10]], firstPerson = 1
Output: [0,1,2,3,5]
Explanation:
At time 0, person 0 shares the secret with person 1.
At time 5, person 1 shares the secret with person 2.
At time 8, person 2 shares the secret with person 3.
At time 10, person 1 shares the secret with person 5.
Thus, people 0, 1, 2, 3, and 5 know the secret after all the meetings.

Example 2:
Input: n = 4, meetings = [[3,1,3],[1,2,2],[0,3,3]], firstPerson = 3
Output: [0,1,3]
Explanation:
At time 0, person 0 shares the secret with person 3.
At time 2, neither person 1 nor person 2 know the secret.
At time 3, person 3 shares the secret with person 0 and person 1.
Thus, people 0, 1, and 3 know the secret after all the meetings.

Example 3:
Input: n = 5, meetings = [[3,4,2],[1,2,1],[2,3,1]], firstPerson = 1
Output: [0,1,2,3,4]
Explanation:
At time 0, person 0 shares the secret with person 1.
At time 1, person 1 shares the secret with person 2, and person 2 shares the secret with person 3.
Note that person 2 can share the secret at the same time as receiving it.
At time 2, person 3 shares the secret with person 4.
Thus, people 0, 1, 2, 3, and 4 know the secret after all the meetings.


Constraints:

2 <= n <= 105
1 <= meetings.length <= 105
meetings[i].length == 3
0 <= xi, yi <= n - 1
xi != yi
1 <= timei <= 105
1 <= firstPerson <= n - 1

"""

import heapq
from collections import defaultdict, deque
from heapq import heappop, heappush
from math import inf
from typing import List

from data_structures.graphs import UnionFind


class FindAllPeopleWitSecret:
    def solve_bfs(self, n: int, meetings: List[List[int]], first_person: int) -> List[int]:
        meeting_graph = defaultdict(list)
        for p1, p2, t in meetings:
            heapq.heappush(meeting_graph[p1], (t, p2))
            heapq.heappush(meeting_graph[p2], (t, p1))

        earliest = [inf] * n  # the earliest time each person learned about the secret
        earliest[0], earliest[first_person] = 0, 0

        to_explore = deque()  # keep (person, time received the secret)
        to_explore.append((0, 0))
        to_explore.append((first_person, 0))

        while to_explore:
            p1, m_time = to_explore.popleft()
            for t, p2 in meeting_graph[p1]:
                if m_time <= t < earliest[p2]:
                    earliest[p2] = t
                    to_explore.append((p2, t))

        return [i for i in range(n) if earliest[i] != inf]

    def solve_dfs(self, n: int, meetings: List[List[int]], first_person: int) -> List[int]:
        meeting_graph = defaultdict(list)
        for p1, p2, t in meetings:
            heapq.heappush(meeting_graph[p1], (t, p2))
            heapq.heappush(meeting_graph[p2], (t, p1))

        earliest = [inf] * n  # the earliest time each person learned about the secret
        earliest[0], earliest[first_person] = 0, 0

        stack = [(0, 0), (first_person, 0)]  # (person, time)
        while stack:
            p1, m_time = stack.pop()
            for t, p2 in meeting_graph[p1]:
                if m_time <= t < earliest[p2]:
                    earliest[p2] = t
                    stack.append((p2, t))
        return [i for i in range(n) if earliest[i] != inf]

    def solve_dfs_recursive(self, n: int, meetings: List[List[int]], first_person: int) -> List[int]:
        # For every person, store the time and label of the person met.
        graph = defaultdict(list)
        for p1, p2, t in meetings:
            graph[p1].append((t, p2))
            graph[p2].append((t, p1))

        # Earliest time at which a person learned the secret
        # as per current state of knowledge. If due to some new information,
        # the earliest time of knowing the secret changes, we will update it
        # and again process all the people whom he/she meets after the time
        # at which he/she learned the secret.
        earliest = [inf] * n
        earliest[0] = 0
        earliest[first_person] = 0

        # Do DFS
        def dfs(person, m_time):
            for t, p2 in graph[person]:
                if m_time <= t < earliest[p2]:
                    earliest[p2] = t
                    dfs(p2, t)

        dfs(0, 0)
        dfs(first_person, 0)

        # Since we visited only those people who know the secret
        # we need to return indices of all visited people.
        return [i for i in range(n) if earliest[i] != inf]

    def solve_earliest_inf_first(self, n: int, meetings: List[List[int]], first_person: int) -> List[int]:
        meeting_graph = defaultdict(list)
        for p1, p2, t in meetings:
            meeting_graph[p1].append((t, p2))
            meeting_graph[p2].append((t, p1))

        # Priority Queue for BFS. It stores (time secret learned, person)
        # It pops the person with the minimum time of knowing the secret.
        to_explore = []  # (time, person)
        heappush(to_explore, (0, 0))
        heappush(to_explore, (0, first_person))

        # Visited array to mark if a person is visited or not.
        # We will mark a person as visited after it is dequeued
        # from the queue.
        visited = [False] * n

        # Do BFS, but pop minimum.
        while to_explore:
            m_time, p1 = heappop(to_explore)
            if visited[p1]:
                continue
            for t, p2 in meeting_graph[p1]:
                if not visited[p2] and t >= m_time:
                    heappush(to_explore, (t, p2))
            visited[p1] = True

        # Since we visited only those people who know the secret
        # we need to return indices of all visited people.
        return [i for i in range(n) if visited[i]]

    def solve_bfs_time_scale(self, n: int, meetings: List[List[int]], first_person: int) -> List[int]:
        # Sort meetings in increasing order of time
        meetings.sort(key=lambda x: x[2])

        # Group Meetings in increasing order of time
        same_time_meetings = defaultdict(list)
        for p1, p2, t in meetings:
            same_time_meetings[t].append((p1, p2))

        # Boolean Array to mark if a person knows the secret or not
        knows_secret = [False] * n
        knows_secret[0] = True
        knows_secret[first_person] = True

        # Process in increasing order of time
        for t in same_time_meetings:
            # For each person, save all the people whom he/she meets at time t
            meet = defaultdict(list)
            for p1, p2 in same_time_meetings[t]:
                meet[p1].append(p2)
                meet[p2].append(p1)

            # Start traversal from those who already know the secret at time t
            # Set to avoid redundancy
            q = set()
            for p1, p2 in same_time_meetings[t]:
                if knows_secret[p1]:
                    q.add(p1)
                if knows_secret[p2]:
                    q.add(p2)

            # Do BFS
            q = deque(q)
            while q:
                person = q.popleft()
                for next_person in meet[person]:
                    if not knows_secret[next_person]:
                        knows_secret[next_person] = True
                        q.append(next_person)

        # List of people who know the secret
        return [i for i in range(n) if knows_secret[i]]

    def solve_union_find_reset(self, n: int, meetings: List[List[int]], first_person: int) -> List[int]:
        # Sort meetings in increasing order of time
        meetings.sort(key=lambda x: x[2])

        # Group Meetings in increasing order of time
        same_time_meetings = defaultdict(list)
        for p1, p2, t in meetings:
            same_time_meetings[t].append((p1, p2))

        # Create graph
        graph = UnionFind(n)
        graph.unite(first_person, 0)

        # Process in increasing order of time
        for t in same_time_meetings:
            # Unite two persons taking part in a meeting
            for p1, p2 in same_time_meetings[t]:
                graph.unite(p1, p2)

            # If any one knows the secret, both will be connected to 0.
            # If no one knows the secret, then reset.
            for p1, p2 in same_time_meetings[t]:
                if not graph.connected(p1, 0):
                    # No need to check for y since x and y were united
                    graph.reset(p1)
                    graph.reset(p2)

        # Al those who are connected to 0 will know the secret
        return [i for i in range(n) if graph.connected(i, 0)]
