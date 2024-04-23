"""
https://leetcode.com/problems/minimum-height-trees/description

310. Minimum Height Trees
Medium

A tree is an undirected graph in which any two vertices are connected by exactly one path.
In other words, any connected graph without simple cycles is a tree.

Given a tree of n nodes labelled from 0 to n - 1, and an array of n - 1 edges
where edges[i] = [ai, bi] indicates that there is an undirected edge between
the two nodes ai and bi in the tree, you can choose any node of the tree as the root.
When you select a node x as the root, the result tree has height h.
Among all possible rooted trees, those with minimum height (i.e. min(h))
are called minimum height trees (MHTs).

Return a list of all MHTs' root labels. You can return the answer in any order.

The height of a rooted tree is the number of edges on the longest downward path
between the root and a leaf.



Example 1:
Input: n = 4, edges = [[1,0],[1,2],[1,3]]
Output: [1]
Explanation: As shown, the height of the tree is 1 when the root is the node with label 1 which is the only MHT.

Example 2:
Input: n = 6, edges = [[3,0],[3,1],[3,2],[3,4],[5,4]]
Output: [3,4]


Constraints:

1 <= n <= 2 * 10**4
edges.length == n - 1
0 <= ai, bi < n
ai != bi
All the pairs (ai, bi) are distinct.
The given input is guaranteed to be a tree and there will be no repeated edges.

"""

from collections import defaultdict
from heapq import heappush
from typing import List


def find_min_height_trees_dfs(n: int, edges: List[List[int]]) -> List[int]:
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    def _dfs(root) -> int:
        stack = [(root, 0)]
        visited = set()
        max_h = 0
        while stack:
            node, curr_h = stack.pop()
            if node not in visited:
                visited.add(node)
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        stack.append((neighbor, curr_h + 1))
            max_h = max(max_h, curr_h)
        return max_h

    max_heap = []
    for root in graph:
        h = _dfs(root)
        if not max_heap or max_heap[0][0] < -h:
            max_heap = [(-h, root)]
        elif max_heap[0][0] == -h:
            heappush(max_heap, (-h, root))

    return [root for _, root in max_heap]


def find_min_height_trees_removing_leaves(n: int, edges: List[List[int]]) -> List[int]:
    if n == 1:
        return [0]
    if not edges:
        return []

    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    # leaves have indegree equals 1
    leaves = [leaf for leaf in graph if len(graph[leaf]) == 1]
    # prune the tree until we have either 1 or 2 nodes only in the graph
    remaining_nodes = n
    while remaining_nodes > 2:
        remaining_nodes -= len(leaves)
        new_leaves = []
        for leaf in leaves:
            neighbor = graph[leaf].pop()
            graph[neighbor].remove(leaf)
            if len(graph[neighbor]) == 1:
                new_leaves.append(neighbor)
        leaves = new_leaves
    return leaves


if __name__ == "__main__":
    cases = [
        (4, [[1, 0], [1, 2], [1, 3]]),
        (6, [[3, 0], [3, 1], [3, 2], [3, 4], [5, 4]]),
        (10, [[0, 3], [1, 3], [2, 3], [4, 3], [5, 3], [4, 6], [4, 7], [5, 8], [5, 9]]),
    ]
    for n, edges in cases:
        print(find_min_height_trees_dfs(n, edges))
        print(find_min_height_trees_removing_leaves(n, edges))
        print()
