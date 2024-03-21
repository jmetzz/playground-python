from collections import deque
from heapq import heappop, heappush
from math import inf
from typing import Any, Dict, List, Optional, Self, Set, Tuple, TypeAlias

from data_structures.graph_utils import build_digraph, plot_digraph

# Alias for an undirected graph
Graph: TypeAlias = Dict[int, Set[int]]

# Alias for a directed graph
WeightedGraph: TypeAlias = Dict[int, Set[Tuple[int, int]]]


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


def traversal_dfs_iterative(graph: Graph, source: int) -> List[int]:
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


def traversal_dfs_recursive(graph: Graph, source: int) -> List[int]:
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


def traversal_bfs(graph: Graph, source: int) -> List[int]:
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


def has_path_bfs_recursive(graph: Graph, source: int, target: int) -> bool:
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


def has_path_bfs_iterative(graph: Graph, source: int, target: int) -> bool:
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


def path_undirected(graph: Graph, source: int, target: int) -> Optional[List[int]]:
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


def count_components_iterative(graph: Graph) -> int:
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


def count_components_recursive(graph: Graph) -> int:
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


def largest_component_size(graph: Graph) -> int:
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


def largest_component_recursive(graph: Graph) -> List[int]:
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


def largest_component_iterative(graph: Graph) -> List[int]:
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


def shortest_path_lenght(graph: Graph, source: int, target: int) -> int:
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


def shortest_path(graph: Graph, source: int, target: int) -> Optional[Tuple[List[int], int]]:
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


def shortest_path_cost_dijkstra(graph: WeightedGraph, source: int, target: int) -> int:
    distances = {vertex: inf for vertex in graph}
    distances[source] = 0
    visited = set()
    min_heap = [(0, source)]
    while min_heap:
        curr_dist, curr = heappop(min_heap)
        visited(curr)
        if curr == target:
            return curr_dist

        for neighbor, w in graph[curr]:
            if neighbor in visited:
                continue
            dist = distances[neighbor]
            if curr_dist + w < dist:
                distances[neighbor] = curr_dist + w
            # remove the neighbor entry from the min_heap
            heappush(min_heap, (distances[neighbor], neighbor))
    return -1


def vertex_degree(graph: Graph, vertex: int) -> int:
    if vertex not in graph:
        raise ValueError(f"Invalid vertex {vertex}")
    return len(graph[vertex])


def max_degree(graph: Graph) -> int:
    max_val = 0
    for neighbors in graph.values():
        max_val = max(len(neighbors), max_val)
    return max_val


def avg_degree(graph: Graph) -> float:
    return sum([len(neighbors) for _, neighbors in graph.items()]) / len(graph)


def number_self_loops(graph: Graph) -> int:
    value = 0
    for node, neighbors in graph.items():
        if node in neighbors:
            value += 1
    return value


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


def topological_order(digraph: Graph) -> List[int]:
    raise NotImplementedError()


def has_cycle_recursive(digraph: Graph) -> bool:
    """In the context of detecting cycles in directed graphs,
    distinguishing between global visitation and path-specific
    visitation is crucial. The visited set alone might not suffice
    for accurately detecting cycles, especially in directed graphs,
    because:

    * Visited Set: A globally maintained visited set helps avoid
    re-exploring nodes already encountered in any traversal,
    optimizing the algorithm by preventing redundant checks.
    However, it does not provide information about the traversal path
    that led to each node, which is essential for detecting cycles.

    * Ancestors Tracking (Path-Specific Visitation): To detect cycles,
    you need to know whether a node has been visited during the
    current traversal path. If you encounter a node that has already
    been visited in the current path, it indicates a cycle
    (a back edge in DFS terms). This cycle detection is not possible
    with just a global visited set because it cannot differentiate
    between visiting a node in the current path versus having visited it
    in a separate, earlier traversal.
    """

    def _dfs(node, ancestors):
        if node in visited:
            # Node has already been visited, no need to traverse again,
            # but doesn't indicate a cycle by itself.
            return False
        if node in ancestors:
            # Node is in the current path, cycled detected
            return True

        ancestors.add(node)
        for neighbor in digraph.get(node, []):
            # return default empty list to safe guard for node with no neighbors
            if _dfs(neighbor, ancestors):
                # cycle detected, propagate back
                return True
        ancestors.remove(node)
        visited.add(node)
        return False

    visited = set()
    for node in digraph:
        if node not in visited:  # noqa: SIM102
            if _dfs(node, set()):
                return True  # cycle detected
    return False


def has_cycle_iterative(digraph: Graph) -> bool:
    WHITE, GRAY, BLACK = 0, 1, 2  # Node states: unvisited, visiting, visited
    state = {node: WHITE for node in digraph}  # Initialize all nodes as unvisited

    def _dfs_seach_cycle(node):
        # every node in the same component should be GRAY,
        # which represent visiting
        state[node] = GRAY  # Mark node as being visited (in the current path)
        for neighbor in digraph.get(node, []):
            if state[neighbor] == GRAY:  # Back edge found, indicating a cycle
                return True
            if state[neighbor] == WHITE and _dfs_seach_cycle(neighbor):
                return True
        state[node] = BLACK  # Mark node as fully visited (exited the path)
        return False

    for node in digraph:
        if state[node] == WHITE:  # noqa: SIM102
            # Unvisited node
            if _dfs_seach_cycle(node):
                return True
    return False


def is_dag(digraph: Graph) -> bool:
    return not has_cycle_recursive(digraph)


def get_any_cycle_path(digraph: Graph) -> List[int]:
    raise NotImplementedError()


if __name__ == "__main__":
    graph, _ = build_digraph(5, [[0, 1], [2, 3], [3, 4], [4, 2]])
    plot_digraph(graph)
    print(has_cycle_recursive(graph))
