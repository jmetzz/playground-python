from collections import deque
from typing import Any, Dict, List, Optional, Self, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx


class UnionFind:
    def __init__(self, nodes: int):
        # Initialize parent and rank arrays.
        # Each node is its own parent
        self.parent = [i for i in range(nodes)]
        self.rank = [0] * nodes
        self.components_count = nodes

    def find(self, x: int) -> int:
        # Find the parent of node x. Use Path Compression
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def unite(self, a: int, b: int) -> None:
        # Unite two nodes x and y, if they are not already united
        p_a = self.find(a)
        p_b = self.find(b)
        if p_a != p_b:
            # Union by Rank Heuristic
            if self.rank[p_a] < self.rank[p_b]:
                self.parent[p_a] = p_b
                self.rank[p_b] += self.rank[p_a]
            else:
                self.parent[p_b] = p_a
                self.rank[p_a] += self.rank[p_b]
            self.components_count -= 1

    def connected(self, x: int, y: int) -> bool:
        # Check if two nodes x and y are connected or not
        return self.find(x) == self.find(y)

    def reset(self, x: int) -> None:
        # Reset the initial properties of node x
        self.parent[x] = x
        self.rank[x] = 0
        self.components_count += 1


class MultiLinkNode:
    def __init__(self, val: Any = None):
        self.val = val
        self.neighbors = set()

    def connect(self, other: Self) -> None:
        self.neighbors.add(other)

    def disconnect(self, other: Self) -> None:
        if other in self.neighbors:
            self.neighbors.remove(other)


def dfs(node: MultiLinkNode, target: Any, visited: Set[MultiLinkNode]) -> bool:
    if node is None:
        return False

    if node.val == target:
        return True

    for n in node.neighbors:
        if n in visited:
            continue  # skip already visited nodes to avoid cycles
        visited.add(n)
        if dfs(n, target, visited):
            return True
    return False


def build_ugraph(num_vertices: int, edges: List[List[int]]) -> Dict[int, Set[int]]:
    """
    Builds an undirected graph represented as an adjacency list.

    Args:
        num_vertices (int): The number of vertices in the graph.
        edges (List[List[int]]): A list of edges where each edge is
        represented as a list [src, neighbor].

    Returns:
        Dict[int, List[int]]: A dictionary representing the undirected graph.
        Each key is a vertex, and each value is a list of adjacent vertices (neighbors).

    Example:
        >>> build_ugraph(3, [[0, 1], [1, 2], [2, 0]])
        {0: [1, 2], 1: [0, 2], 2: [1, 0]}
    """
    graph = {vertex: set() for vertex in range(num_vertices)}
    for src, neighbor in edges:
        graph[src].add(neighbor)
        graph[neighbor].add(src)
    return graph


def build_digraph(
    num_vertices: int, edges: List[List[int]], calc_indegree: bool = False
) -> Tuple[Dict[int, Set[int]], Optional[Dict[int, int]]]:
    """
    Builds a directed graph represented as an adjacency list,
    with the option to calculate indegrees.

    Args:
        num_vertices (int): The number of vertices in the graph.
        edges (List[List[int]]): A list of directed edges where each edge is
        represented as a list [src, destination].
        calc_indegree (bool, optional): If True, also calculates and returns the
        indegree for each vertex. Defaults to False.

    Returns:
        Tuple[Dict[int, List[int]], Optional[Dict[int, int]]]: A tuple containing the graph as a dictionary
        and optionally a dictionary of indegrees for each vertex.
        If calc_indegree is False, the second element of the tuple is None.

    Example:
        >>> build_digraph(3, [[0, 1], [1, 2]], True)
        ({0: [1], 1: [2], 2: []}, {1: 1, 2: 1})
    """
    graph = {vertex: set() for vertex in range(num_vertices)}
    indegree = {vertex: 0 for vertex in range(num_vertices)} if calc_indegree else None

    for src, neighbor in edges:
        graph[src].add(neighbor)
        if calc_indegree:
            indegree[neighbor] += 1
    return (graph, indegree) if calc_indegree else (graph, None)


def dfs_graph_traversal_iterative(graph: Dict[int, Set[int]], source: int) -> List[int]:
    stack = []
    visited = set()
    stack.append(source)
    traversal = []
    while stack:
        node = stack.pop()
        if node not in visited:
            traversal.append(node)
            visited.add(node)
            for neighbor in graph[node]:
                stack.append(neighbor)

    return traversal


def dfs_graph_traversal_recursive(graph: Dict[int, Set[int]], source: int) -> List[int]:
    traversal = []
    visited = set()

    def _dfs(node: int):
        if node in visited:
            return

        visited.add(node)
        traversal.append(node)
        for neighbor in graph[node]:
            _dfs(neighbor)

    _dfs(source)
    return traversal


def bfs_graph_traversal(graph: Dict[int, Set[int]], source: int) -> List[int]:
    traversal = []
    visited = set()

    explore_queue = deque([source])

    while explore_queue:
        node = explore_queue.popleft()
        if node not in visited:
            visited.add(node)
            traversal.append(node)
            for neighbor in graph[node]:
                explore_queue.append(neighbor)

    return traversal


def has_path_bfs_recursive(graph: Dict[int, Set[int]], source: int, target: int) -> bool:
    if len(graph) == 0:
        return False
    visited = set()

    def _dfs(node):
        if node == target:
            return True

        if node in visited:
            return False

        visited.add(node)
        for neighbor in graph[node]:  # noqa: SIM110
            if _dfs(neighbor):
                return True

        return False

    return _dfs(source)


def has_path_bfs_iterative(graph: Dict[int, Set[int]], source: int, target: int) -> bool:
    if len(graph) == 0:
        return False
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


def undirected_path(graph: Dict[int, Set[int]], source: int, target: int) -> Optional[List[int]]:
    """Traverse from source to target collecting all nodes along the way

    To correctly find a path (if one exists) from the source to the target,
    a recursive depth-first search (DFS) can be more suitable, since reconstructing
    the actual path from the source to the target is done while backtracking.
    """

    def _dfs(node, current_path) -> List[int]:
        if node == target:
            return current_path + [node]
        if node in visited:
            # continuing would get us in a cycle
            return None

        visited.add(node)
        for neighbor in graph[node]:
            intermediate_path = _dfs(neighbor, current_path + [node])
            if intermediate_path:
                # the target node was found recursivelly
                return intermediate_path
        # is the loop is done without returning the intermediate path,
        # the path from the current path does not lead to the target node
        return None

    visited = set()
    return _dfs(source, [])


def count_components_iterative(graph: Dict[int, Set[int]]) -> int:
    def _dfs_traverse_from(node):
        stack = [node]
        while stack:
            curr_node = stack.pop()
            if curr_node not in visited:
                visited.add(curr_node)
                stack.extend(graph[curr_node])
        return

    visited = set()
    count = 0
    for node in graph:
        if node not in visited:
            _dfs_traverse_from(node)
            count += 1
    return count


def count_components_recursive(graph: Dict[int, Set[int]]) -> int:
    def _dfs_traverse_from(node):
        if node in visited:
            return False

        visited.add(node)
        for neighbor in graph[node]:
            _dfs_traverse_from(neighbor)
        return True

    visited = set()
    count = 0
    for node in graph:
        if _dfs_traverse_from(node):
            count += 1
    return count


def largest_component_size(graph: Dict[int, Set[int]]) -> int:
    def _dfs_traverse_from(node):
        if node in visited:
            return 0

        visited.add(node)
        size = 1
        for neighbor in graph[node]:
            size += _dfs_traverse_from(neighbor)
        return size

    visited = set()
    max_size = 0
    for node in graph:
        curr_size = _dfs_traverse_from(node)
        if curr_size > max_size:
            max_size = curr_size
    return max_size


def largest_component_recursive(graph: Dict[int, Set[int]]) -> List[int]:
    def _dfs_traverse_from(node):
        if node in visited:
            return []
        visited.add(node)
        component = [node]
        for neighbor in graph[node]:
            component += _dfs_traverse_from(neighbor)
        return component

    visited = set()
    largest_component = []

    for node in graph:
        curr_component = _dfs_traverse_from(node)
        if len(curr_component) > len(largest_component):
            largest_component = curr_component
    return largest_component


def largest_component_iterative(graph: Dict[int, Set[int]]) -> List[int]:
    visited = set()
    largest_component = []

    def _dfs_traverse_from(node):
        stack = [node]
        component = []
        while stack:
            curr_node = stack.pop()
            if curr_node not in visited:
                visited.add(curr_node)
                component.append(curr_node)
                stack.extend(graph[curr_node] - visited)  # Only add unvisited neighbors

        return component

    for node in graph:
        curr_component = _dfs_traverse_from(node)
        if len(curr_component) > len(largest_component):
            largest_component = curr_component
    return largest_component


def lenght_shortest_path(graph: Dict[int, Set[int]], source: int, target: int) -> int:
    visited = set()
    queue = deque([(source, 1)])  # store the node and the path lenght from source
    while queue:
        node, dist = queue.popleft()
        if node == target:
            return dist
        if node not in visited:
            visited.add(node)
            for neighbor in graph[node]:
                queue.append((neighbor, dist + 1))
    return 0


def shortest_path(graph: Dict[int, Set[int]], source: int, target: int) -> Optional[Tuple[List[int], int]]:
    visited = set([source])
    parent = {source: None}  # Track the parent of each node for path reconstruction

    queue = deque([(source, 0)])  # Store the node and the path length from source
    while queue:
        node, dist = queue.popleft()
        if node == target:
            # Reconstruct the path from source to target
            node_sequence = []
            while node is not None:
                node_sequence.append(node)
                node = parent[node]
            return node_sequence[::-1], dist  # Return reversed path and distance

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = node  # Set the current node as parent of neighbor
                queue.append((neighbor, dist + 1))

    return None, 0  # Return None if no path exists


def plot_graph(graph: Dict[int, Set[int]]) -> None:
    """
    Visualizes the given graph

    Each key is a node and its value is a set of connected nodes.

    Args:
    - graph_dict: A dictionary representing the graph's adjacency list,
                  where each key is a node and the value is a set of nodes
                  to which it is connected.
    """
    G = nx.Graph()

    # Add nodes and edges from the graph_dict
    for node, neighbors in graph.items():
        G.add_node(node)  # Ensure the node is added even if it has no neighbors
        for neighbor in neighbors:
            G.add_edge(node, neighbor)

    nx.draw(
        G,
        with_labels=True,
        node_color="skyblue",
        node_size=700,
        edge_color="k",
        linewidths=1,
        font_size=15,
        pos=nx.spring_layout(G, seed=42),
    )

    plt.show()


if __name__ == "__main__":
    # g = build_ugraph(6, [[0, 1], [0, 2], [1, 4], [2, 4], [1, 3], [3, 5]])
    # print(dfs_graph_traversal_recursive(g, 0))
    # print(dfs_graph_traversal_iterative(g, 0))
    # print(bfs_graph_traversal(g, 0))
    # print("-----")
    # dg, _ = build_digraph(6, [[0, 1], [0, 2], [1, 4], [2, 4], [1, 3], [3, 5]])
    # print(dfs_graph_traversal_recursive(dg, 0))
    # print(dfs_graph_traversal_iterative(dg, 0))
    # print(bfs_graph_traversal(dg, 0))
    # print(bfs_graph_traversal(dg, 3))

    # graph = build_ugraph(10, [[0, 7], [0, 8], [6, 1], [2, 0], [0, 4], [5, 8], [4, 7], [1, 3], [3, 5], [6, 5]])
    # print(has_path_bfs_recursive(graph, 7, 5))
    # print(has_path_bfs_iterative(graph, 7, 5))

    # plot_graph(graph)
    # graph = build_ugraph(7, [[0, 1], [0,2], [1,2], [2, 3], [2, 4], [5, 6]])
    # plot_graph(graph)

    # print(undirected_path(graph, 1, 3))

    graph = build_ugraph(8, [[0, 1], [3, 5], [4, 5], [5, 6], [5, 7]])

    print(count_components_iterative(graph))
    print(count_components_recursive(graph))
    print(largest_component_size(graph))
    print(largest_component_recursive(graph))
    print(largest_component_iterative(graph))
    print("shortest path")
    print(lenght_shortest_path(graph, 3, 6))
    print(lenght_shortest_path(graph, 0, 1))
    print(lenght_shortest_path(graph, 0, 6))
    # plot_graph(graph)

    # plot_graph({0: {1}, 1: {2, 3, 4}, 2: {1, 5}, 3: {1}, 4: {1, 5}, 5: {2, 4}} )
    # plot_graph({0: {1}, 1: {0}, 2: {3}, 3: {2}} )

    GRAPH_SIMPLE = {0: {1, 2}, 1: {0, 3}, 2: {0}, 3: {1}}  # Simple connected graph
    GRAPH_CYCLIC = {0: {1}, 1: {2}, 2: {0}}  # Cyclic graph (triangle)
    GRAPH_COMPLEX = {0: {1}, 1: {2, 3, 4}, 2: {1, 5}, 3: {1}, 4: {1, 5}, 5: {2, 4}}  # More complex graph
    GRAPH_SMALL_DISCONNECTED = {0: {1}, 1: {0}, 2: {3}, 3: {2}}  # Disconnected graph (two components)
    GRAPH_BIG_DISCONNECTED = {0: {1}, 1: {0}, 2: {}, 3: {5}, 4: {5}, 5: {3, 4, 6, 7}, 6: {5}, 7: {5}}

    GRAPHS = [GRAPH_SIMPLE, GRAPH_CYCLIC, GRAPH_COMPLEX, GRAPH_SMALL_DISCONNECTED, GRAPH_BIG_DISCONNECTED]
    for graph in GRAPHS:
        plot_graph(graph)
