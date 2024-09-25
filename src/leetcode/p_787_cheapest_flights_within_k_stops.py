"""https://leetcode.com/problems/cheapest-flights-within-k-stops/description

787. Cheapest Flights Within K Stops
Medium

There are n cities connected by some number of flights. You are given an array flights
where `flights[i] = [fromi, toi, pricei]` indicates that there is a flight from
city fromi to city toi with cost pricei.

You are also given three integers src, dst, and k, return the cheapest price
from src to dst with at most k stops. If there is no such route, return -1.



Example 1:
Input: n = 4, flights = [[0,1,100],[1,2,100],[2,0,100],[1,3,600],[2,3,200]], src = 0, dst = 3, k = 1
Output: 700
Explanation:
The graph is shown above.
The optimal path with at most 1 stop from city 0 to 3 is marked in red and has cost 100 + 600 = 700.
Note that the path through cities [0,1,2,3] is cheaper but is invalid because it uses 2 stops.

Example 2:
Input: n = 3, flights = [[0,1,100],[1,2,100],[0,2,500]], src = 0, dst = 2, k = 1
Output: 200
Explanation:
The graph is shown above.
The optimal path with at most 1 stop from city 0 to 2 is marked in red and has cost 100 + 100 = 200.

Example 3:
Input: n = 3, flights = [[0,1,100],[1,2,100],[0,2,500]], src = 0, dst = 2, k = 0
Output: 500
Explanation:
The graph is shown above.
The optimal path with no stops from city 0 to 2 is marked in red and has cost 500.


Constraints:

1 <= n <= 100
0 <= flights.length <= (n * (n - 1) / 2)
flights[i].length == 3
0 <= fromi, toi < n
fromi != toi
1 <= pricei <= 104
There will not be any multiple flights between two cities.
0 <= src, dst, k < n
src != dst
"""

from collections import defaultdict, deque


class CheapestFlights:
    def solve(self, n: int, flights: list[list[int]], src: int, dst: int, k: int) -> int:
        adjacent = defaultdict(list)
        for f in flights:
            adjacent[f[0]].append((f[1], f[2]))  # append destination and associated cost

        cost_to_reach = [float("inf")] * n
        cost_to_reach[src] = 0

        frontier = deque()
        frontier.append((src, 0))  # (node, distance) from src
        steps = 0

        while frontier and steps <= k:
            qtt_to_explore = len(frontier)
            for _ in range(qtt_to_explore):
                # ensure only the adjacent nodes at 1-step distance are explored
                curr_node, curr_cost = frontier.popleft()
                if curr_node not in adjacent:
                    continue

                # explore all adjacent nodes
                for adj_node, step_cost in adjacent[curr_node]:
                    if curr_cost + step_cost >= cost_to_reach[adj_node]:
                        continue
                    cost_to_reach[adj_node] = curr_cost + step_cost
                    frontier.append((adj_node, cost_to_reach[adj_node]))
            steps += 1
        return cost_to_reach[dst] if cost_to_reach[dst] != float("inf") else -1
