"""1971. Find if Path Exists in Graph
Easy

There is a bi-directional graph with n vertices, where each vertex is labeled
from 0 to n - 1 (inclusive). The edges in the graph are represented as a
2D integer array edges, where each edges[i] = [ui, vi] denotes a bi-directional edge
between vertex ui and vertex vi. Every vertex pair is connected by at most one edge,
and no vertex has an edge to itself.

You want to determine if there is a valid path that exists
from vertex source to vertex destination.

Given edges and the integers n, source, and destination, return true if there is
a valid path from source to destination, or false otherwise.

Example 1:
Input: n = 3, edges = [[0,1],[1,2],[2,0]], source = 0, destination = 2
Output: true
Explanation: There are two paths from vertex 0 to vertex 2:
- 0 → 1 → 2
- 0 → 2

Example 2:
Input: n = 6, edges = [[0,1],[0,2],[3,5],[5,4],[4,3]], source = 0, destination = 5
Output: false
Explanation: There is no path from vertex 0 to vertex 5.


Constraints:

1 <= n <= 2 * 105
0 <= edges.length <= 2 * 105
edges[i].length == 2
0 <= ui, vi <= n - 1
ui != vi
0 <= source, destination <= n - 1
There are no duplicate edges.
There are no self edges.
"""

from typing import Dict, Set


def has_path_bfs_recursive(graph: Dict[int, Set[int]], source: int, target: int) -> bool:
    visited = set()

    def _dfs(node):
        if node in visited:
            return False

        if node == target:
            return True

        visited.add(node)
        for neighbor in graph[node]:  # noqa: SIM110
            if _dfs(neighbor):
                return True
        return False

    return _dfs(source)


def has_path_bfs_iterative(graph: Dict[int, Set[int]], source: int, target: int) -> bool:
    visited = set()
    stack = []
    stack.append(source)
    while stack:
        curr_node = stack.pop()
        if curr_node == target:
            return True
        if curr_node not in visited:
            visited.add(curr_node)
            for neighbor in graph[curr_node]:
                stack.append(neighbor)

    return False
