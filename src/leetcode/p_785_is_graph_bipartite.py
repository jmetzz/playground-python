"""
https://leetcode.com/problems/is-graph-bipartite/description/

785. Is Graph Bipartite?
Medium

There is an undirected graph with n nodes, where each node is numbered between 0 and n - 1.
You are given a 2D array graph, where graph[u] is an array of nodes that node u is adjacent to.
More formally, for each v in graph[u], there is an undirected edge between node u and node v.
The graph has the following properties:

There are no self-edges (graph[u] does not contain u).
There are no parallel edges (graph[u] does not contain duplicate values).
If v is in graph[u], then u is in graph[v] (the graph is undirected).
The graph may not be connected, meaning there may be two nodes u and v
such that there is no path between them.
A graph is bipartite if the nodes can be partitioned into two independent sets A and B
such that every edge in the graph connects a node in set A and a node in set B.

Return true if and only if it is bipartite.


Example 1:
Input: graph = [[1,2,3],[0,2],[0,1,3],[0,2]]
Output: false
Explanation: There is no way to partition the nodes into two independent sets such that
every edge connects a node in one and a node in the other.

Example 2:
Input: graph = [[1,3],[0,2],[1,3],[0,2]]
Output: true
Explanation: We can partition the nodes into two sets: {0, 2} and {1, 3}.


Constraints:

graph.length == n
1 <= n <= 100
0 <= graph[u].length < n
0 <= graph[u][i] <= n - 1
graph[u] does not contain u.
All the values of graph[u] are unique.
If graph[u] contains v, then graph[v] contains u.

"""


from collections import deque

from data_structures.graphs import Graph


def is_bipartite_dfs(graph: Graph) -> bool:
    """Can the vertices of the given graph be assigned one of two colors
    in such a way that no edge connects vertices of the same color?
    """

    def _dfs_traverse(node, color):
        """
        traverse the graph, attempting to assign alternating colors
        to neighboring vertices. If at any point it finds two adjacent
        vertices with the same color, it concludes the graph is not bipartite.
        """
        if node in color_map:
            return color_map[node] == color
        color_map[node] = color
        return all(_dfs_traverse(neighbor, not color) for neighbor in graph[node])

    color_map = {}

    for node in graph:
        if node not in color_map:  # noqa: SIM102
            # it's critical to ensure that the first node in each disconnected component
            # of the graph is explicitly assigned a starting color when the traversal begins.
            # Assume False as starting "color".
            if not _dfs_traverse(node, False):
                return False  # Early stop if a component is not bipartite
    return True


def is_bipartite_bfs(graph: Graph) -> bool:
    """ "
    Do a BFS starting from any arbitrary node.
    Assign it to group 1. Assign each neighbor to group 2.
    Return False if any of our neighbor is already assigned to group 1.
    Otherwise continue the bfs with group = other group
    """

    def _bfs_traverse(start_node):
        queue = deque([start_node])
        color_map[start_node] = False
        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if neighbor in color_map:
                    if color_map[neighbor] == color_map[node]:
                        return False
                else:
                    queue.append(neighbor)
                    color_map[neighbor] = not color_map[node]
        return True

    color_map = {}
    for node in graph:  # noqa: SIM110
        if node not in color_map and not _bfs_traverse(node):
            return False
    return True
