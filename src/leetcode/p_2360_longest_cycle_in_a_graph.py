"""
https://leetcode.com/problems/longest-cycle-in-a-graph/description/

2360. Longest Cycle in a Graph
Hard

You are given a directed graph of n nodes numbered from 0 to n - 1,
where each node has at most one outgoing edge.

The graph is represented with a given 0-indexed array edges of size n,
indicating that there is a directed edge from node i to node edges[i].
If there is no outgoing edge from node i, then edges[i] == -1.

Return the length of the longest cycle in the graph. If no cycle exists,
return -1.

A cycle is a path that starts and ends at the same node.



Example 1:
Input: edges = [3,3,4,2,3]
Output: 3
Explanation: The longest cycle in the graph is the cycle: 2 -> 4 -> 3 -> 2.
The length of this cycle is 3, so 3 is returned.

Example 2:
Input: edges = [2,-1,3,1]
Output: -1
Explanation: There are no cycles in this graph.

Constraints:
n == edges.length
2 <= n <= 105
-1 <= edges[i] < n
edges[i] != i
"""

from typing import List

from data_structures.graph_utils import Graph, transpose_digraph


def longest_cycle_set(edges: List[int]) -> int:
    """
    Finds the length of the longest cycle in a directed graph.

    Args:
        edges (List[int]): A list where the value at each index i represents the
                            successor of vertex i, or -1 if i has no successor.

    Returns:
        int: The length of the longest cycle found in the graph. Returns -1 if no cycle is found.

    This implementation utilizes a set to track visited vertices, thereby avoiding the side-effect
    of modifying the original `edges` list. It iterates over each vertex, following the chain of
    successors to detect cycles and measure their lengths, updating the answer with the length of
    the longest cycle found.
    """
    answer = -1
    visited = set()  # Tracks visited vertices without modifying edges

    for vertex in range(len(edges)):
        component = []
        current = vertex
        while edges[current] >= 0 and current not in visited:
            component.append(current)
            visited.add(current)
            current = edges[current]

        if current in component:
            cycle_start_index = component.index(current)
            answer = max(answer, len(component) - cycle_start_index)

    return answer


def longest_cycle_boolean_arr(edges: List[int]) -> int:
    """
    Finds the length of the longest cycle in a directed graph.

    Args:
        edges (List[int]): A list where the value at each index i represents the
                            successor of vertex i, or -1 if i has no successor.

    Returns:
        int: The length of the longest cycle found in the graph. Returns -1 if no cycle is found.

    This function uses a boolean array to track visited vertices, offering a slight performance
    advantage over a set for dense, integer-indexed graphs. It avoids modifying the original `edges`
    list by iterating over vertices, following their chain of successors to detect cycles, and
    calculating their lengths to find the longest cycle.
    """
    answer = -1
    visited = [False] * len(edges)  # Boolean array to track visited vertices

    for vertex in range(len(edges)):
        component = []
        current = vertex
        while edges[current] >= 0 and not visited[current]:
            component.append(current)
            visited[current] = True
            current = edges[current]

        if current in component:
            cycle_start_index = component.index(current)
            answer = max(answer, len(component) - cycle_start_index)

    return answer


def longest_cycle_side_effect(edges: List[int]) -> int:
    """
    Finds the length of the longest cycle in a directed graph

    where edges is a list such that edges[i] is the successor of vertex i,
    or -1 if i has no successor. This representation simplifies the graph to
    a series of directed chains and cycles, rather than using an adjacency list
    or matrix.

    Args:
        edges (List[int]): A list where the value at each index i represents the
                            successor of vertex i, or -1 if i has no successor.

    Returns:
        int: The length of the longest cycle found in the graph. Returns -1 if no cycle is found.


    NOTE: this function operates on side-effect! it does modify the input array.

    Here's a breakdown of how it works:

    1. The function iterates over each vertex in the graph, and attempts to
    follow the chain of successors until it either reaches a vertex without
    a successor (edges[vertex] == -1) or encounters a vertex already seen
    in the current chain.

    2. For each vertex, it builds a component list that tracks the sequence
    of vertices visited starting from the current vertex. This sequence
    includes both the chain leading up to a cycle and the cycle itself,
    if one is encountered.

    3. As it visits vertices, it marks them as visited by setting
    edges[vertex] = -1. This prevents revisiting vertices in future iterations
    and effectively removes them from the graph.

    4. A cycle is detected if, during the traversal, it encounters a vertex that
    is already in the component list. The start of the cycle within component is
    found using component.index(vertex), and the length of the cycle is
    calculated as len(component) - component.index(vertex).

    """
    answer = -1
    for vertex in range(len(edges)):
        component = []
        current = vertex
        while edges[current] >= 0:
            component.append(current)
            successor = edges[current]  # gets the reference to the next node to visit
            edges[current] = -1  # mark the vertex as seen
            current = successor

        if current in component:
            cycle_length = len(component) - component.index(current)
            answer = max(answer, cycle_length)
    return answer


def longest_cycle_kosaraju(edges: List[int]) -> int:
    """
    Finds the length of the longest cycle in a directed graph without modifying the input list.

    Args:
        edges (List[int]): A list where the value at each index i represents the
                            successor of vertex i, or -1 if i has no successor.

    Returns:
        int: The length of the longest cycle found in the graph. Returns -1 if no cycle is found.

    This function uses the Kosaraju's algorithm identifies SCCs through two DFS passes.
    It is a nice and elegant algorithm, but to slow for this problem.
    """

    num_vertices = len(edges)
    graph = {vertex: set() for vertex in range(num_vertices)}

    for idx, neighbor in enumerate(edges):
        if neighbor != -1:
            graph[idx].add(neighbor)

    visited = set()
    order = []  # Stack to store the vertices based on their finishing times in DFS

    def _dfs(graph: Graph, vertex: int, action: callable) -> None:
        # Perform DFS and apply the action
        # (e.g., fill order or collect SCC) to each visited vertex
        visited.add(vertex)
        for neighbor in graph.get(vertex, []):
            if neighbor not in visited:
                _dfs(graph, neighbor, action)
        action(vertex)

    # First DFS pass to fill order
    #     The action for the first DFS: append vertex to the order stack
    #     This fills the order stack with vertices in decreasing order
    #     of their finishing times
    for vertex in graph:
        if vertex not in visited:
            _dfs(graph, vertex, order.append)

    transposed_graph = transpose_digraph(graph)
    visited.clear()  # Reset visited for the second DFS pass

    # Perform DFS on the transposed graph in the order determined above
    #     Action for the second DFS: append vertex to current SCC.
    #     Since we're working with the last component in sccs list,
    #     this avoids passing the current_scc list around.

    def increment_count(dummy):
        nonlocal count
        count += 1

    max_size = 0
    while order:
        vertex = order.pop()
        if vertex not in visited:
            count = 0
            _dfs(transposed_graph, vertex, increment_count)
            if count > max_size:
                max_size = count

    return max_size if max_size > 1 else -1
