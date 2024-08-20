from collections import deque
from heapq import heappop, heappush
from math import inf
from typing import Any, Dict, List, Optional, Self, Set, Tuple, TypeAlias

from data_structures.graph_utils import build_digraph, calculate_indegree, plot_digraph, transpose_digraph

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


def shortest_path_length(graph: Graph, source: int, target: int) -> int:
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


def topological_order_recursive(digraph: Graph) -> List[int]:
    """
    This is the typical DFS for topological order,
    in which nodes are added to the topological order in reverse post-order.

    Essentially, a node is added to the order after all nodes reachable from it
    have been visited.

    It's important to note that topological sorting is only applicable to DAGs.
    If a cycle is detected during DFS, it indicates that a topological order
    does not exist for the graph.
    Therefore, Directed Acyclic Graph (DAG) is a prerequisite for
    topological sorting.

    """
    order = []
    has_cycle = False
    VISITED = set()

    # Track Exploration State, from the moment a node is encountered
    # until all its descendants have been fully explored
    EXPLORING_COMPONENT = set()

    def dfs_explore(node):
        nonlocal has_cycle
        # node not visited -> start dfs exploration of the component.
        # explores as deeply as possible along each branch before backtracking
        if node in EXPLORING_COMPONENT:
            has_cycle = True
            return

        if node not in VISITED:
            EXPLORING_COMPONENT.add(node)
            # Not all nodes in a directed graph may have outgoing edges
            # therefor, check for neighbor and use default [] is there is none.
            for neighbor in digraph.get(node, []):
                dfs_explore(neighbor)

            # Once the recursive exploration from a node concludes (all nodes
            # reachable from it have been explored), the node is removed from the
            # VISITED_IN_CURRENT_COMPONENT set. It's now safe to say that
            # there are no paths from this node back to itself through its descendants,
            # as all paths have been explored.
            EXPLORING_COMPONENT.remove(node)

            # After fully exploring the node, it can be added to the
            # result list (in reverse post-order) and marked as visited
            # to prevent redundant explorations
            VISITED.add(node)
            order.append(node)

    for node in digraph:
        # outer loop to ensure all components in the digraph are explored
        if node not in VISITED and not has_cycle:
            dfs_explore(node)

    if has_cycle:
        return []  # Cannot topologically sort if there is a cycle

    # reverse the list to return the the propor order
    return order[::-1]


def topological_order_recursive_2(digraph: Graph) -> List[int]:
    """
    This is the typical DFS for topological order,
    in which nodes are added to the topological order in reverse post-order.

    Essentially, a node is added to the order after all nodes reachable from it
    have been visited.

    It's important to note that topological sorting is only applicable to DAGs.
    If a cycle is detected during DFS, it indicates that a topological order
    does not exist for the graph.
    Therefore, Directed Acyclic Graph (DAG) is a prerequisite for
    topological sorting.

    """
    order = []
    VISITED = set()
    EXPLORING_COMPONENT = set()

    def dfs_explore(node):
        if node in EXPLORING_COMPONENT:
            # cycle detected
            return False

        if node not in VISITED:
            EXPLORING_COMPONENT.add(node)
            for neighbor in digraph.get(node, []):
                if not dfs_explore(neighbor):
                    # cycle detected
                    return False
            EXPLORING_COMPONENT.remove(node)
            VISITED.add(node)
            order.append(node)
        # successful exploration without finding a cycle
        return True

    for node in digraph:
        if node not in VISITED:  # noqa: SIM102
            if dfs_explore(node) is False:
                # Cannot topologically sort if there is a cycle
                return []

    # reverse the list to return the the propor order
    return order[::-1]


def topological_order_iterative(digraph: Graph) -> List[int]:
    """
    A topological order only for di-graph with no cycles.

    This approach leverages the concept of in-degree (the number of incoming edges to a node)
    to find the starting points for the sorting and to determine when a node has no
    remaining dependencies and can be added to the topological order.

    It's iterative and avoids recursion, which can be advantageous for large graphs where
    deep recursion might lead to stack overflow.

    Cycle detection is implicit. If the graph contains a cycle, not all nodes
    will be processed, indicated by the final top_order list not containing all
    graph nodes.

    Using a queue ensures that nodes are processed in a level-by-level manner,
    similar to Breadth-First Search (BFS).
    This means that nodes are processed roughly in the order they are discovered.
    """
    # calculate in-degree for each node
    in_degree = calculate_indegree(digraph)

    # Queue for nodes with no incoming edge.
    # These nodes are starting points for the sorting,
    # as they don't depend on any other nodes.
    queue = deque([u for u in in_degree if in_degree[u] == 0])
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)
        # "fake Remove" the node from the di-graph
        # by decreasing in-degree for all neighbors
        for neighbor in digraph.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                # add to the queue all nodes that have no more dependencies
                queue.append(neighbor)

    if len(order) == len(digraph):
        # Check if topological ordering was possible
        return order
    else:
        # Cycle detected or graph is not a DAG
        return []


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

    def _dfs_search_cycle(node):
        # every node in the same component should be GRAY,
        # which represent visiting
        state[node] = GRAY  # Mark node as being visited (in the current path)
        for neighbor in digraph.get(node, []):
            if state[neighbor] == GRAY:  # Back edge found, indicating a cycle
                return True
            if state[neighbor] == WHITE and _dfs_search_cycle(neighbor):
                return True
        state[node] = BLACK  # Mark node as fully visited (exited the path)
        return False

    for node in digraph:
        if state[node] == WHITE:  # noqa: SIM102
            # Unvisited node
            if _dfs_search_cycle(node):
                return True
    return False


def is_dag(digraph: Graph) -> bool:
    raise NotImplementedError()


def get_any_cycle_path(digraph: Graph) -> List[int]:
    raise NotImplementedError()


def kosaraju(graph: Graph) -> List[List[int]]:
    """
    Find all strongly connected components (SCCs) in a directed graph.

    Kosaraju's algorithm identifies SCCs through two DFS passes, leveraging graph edge reversals. It
    runs in linear time, O(V+E), where V and E are the number of vertices and edges, respectively.

    The algorithm's steps are:

    1. First Pass: Perform DFS on the original graph to determine the vertices' exploration order for
    the second pass. Vertices are added to a stack as each DFS finishes, ensuring they're processed
    in decreasing order of their finishing times in the next pass.

    2. Second Pass: Conduct DFS on the transposed graph, exploring vertices in the order they were
    added to the stack. Each DFS discovers all vertices reachable from a start vertex in the
    transposed graph, identifying a SCC.

    The key insight is that in a directed graph, if vertex u is reachable from vertex v, then in the
    transposed graph, v is reachable from u. This property holds for all vertices within a SCC.

    Why post-order DFS?
    - First Pass: vertices are added to a stack in post-order upon completing their DFS exploration.
    Vertices in sink SCCs, which have no exits leading to other SCCs, are among the last explored and
    thus are placed late onto the stack. Consequently, during the second pass on the transposed graph,
    these vertices are processed early. This characteristic ensures that the algorithm efficiently
    identifies SCCs starting with those that are 'sinks' in the context of the graph's SCC structure.
    - Second Pass: Vertices are grouped into components after ensuring all reachable vertices have been
    visited, guaranteeing accurate SCC identification.

    Why not act immediately after `visited.add(vertex)`?
    - Adopting a pre-order traversal by placing the action right after marking a vertex as visited
    disrupts the order for the second DFS pass. Kosaraju's algorithm relies on post-order traversal
    for its efficiency and accuracy.

    Args:
        graph (Graph): The directed graph for finding SCCs, represented as an adjacency list.

    Returns:
        List[List[int]]: Lists of vertices, each representing a strongly connected component.
    """
    visited = set()
    order = []  # Stack to store the vertices based on their finishing times in DFS
    sccs = []  # To store the strongly connected components

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
    while order:
        vertex = order.pop()
        if vertex not in visited:
            sccs.append([])  # Start a new SCC list for this component
            _dfs(transposed_graph, vertex, lambda v: sccs[-1].append(v))

    return sccs


if __name__ == "__main__":
    graph, _ = build_digraph(5, [[0, 1], [2, 3], [3, 4], [4, 2]])
    plot_digraph(graph)
    print(has_cycle_recursive(graph))
