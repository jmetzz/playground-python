import random

import matplotlib.pyplot as plt
import networkx as nx


def build_ugraph(num_vertices: int, edges: list[list[int]]) -> dict[int, set[int]]:
    """Builds an undirected graph represented as an adjacency list.

    Args:
    ----
        num_vertices (int): The number of vertices in the graph.
        edges (List[List[int]]): A list of edges where each edge is
        represented as a list [src, neighbor].

    Returns:
    -------
        Dict[int, List[int]]: A dictionary representing the undirected graph.
        Each key is a vertex, and each value is a list of adjacent vertices (neighbors).

    Example:
    -------
        >>> build_ugraph(3, [[0, 1], [1, 2], [2, 0]])
        {0: [1, 2], 1: [0, 2], 2: [1, 0]}

    """
    graph = {vertex: set() for vertex in range(num_vertices)}
    for src, neighbor in edges:
        graph[src].add(neighbor)
        graph[neighbor].add(src)
    return graph


def build_digraph(
    num_vertices: int, edges: list[list[int]], calc_indegree: bool = False
) -> tuple[dict[int, set[int]], dict[int, int] | None]:
    """Builds a directed graph represented as an adjacency list,
    with the option to calculate indegrees.

    Args:
    ----
        num_vertices (int): The number of vertices in the graph.
        edges (List[List[int]]): A list of directed edges where each edge is
        represented as a list [src, destination].
        calc_indegree (bool, optional): If True, also calculates and returns the
        indegree for each vertex. Defaults to False.

    Returns:
    -------
        Tuple[Dict[int, List[int]], Optional[Dict[int, int]]]: A tuple containing the graph as a dictionary
        and optionally a dictionary of indegrees for each vertex.
        If calc_indegree is False, the second element of the tuple is None.

    Example:
    -------
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


def calculate_indegree(graph: dict[int, set[int]]) -> dict[int, int]:
    """Calculate the in-degree for each vertex in a directed graph.

    The in-degree of a vertex is the number of edges directed towards it. This function
    traverses the given graph and calculates the in-degree for each vertex, ensuring
    that even vertices with an in-degree of 0 are included in the result.

    Args:
    ----
        graph (Graph): A directed graph represented as a dictionary where each key is a
                       vertex and its value is a set of vertices to which there are
                       outgoing edges from the key vertex.

    Returns:
    -------
        Dict[int, int]: A dictionary where each key is a vertex in the graph and its value
                        is the in-degree of that vertex. Vertices with no incoming edges
                        are included in the dictionary with an in-degree of 0.

    Example:
    -------
        >>> graph = {1: {2, 3}, 2: {3}, 3: set()}
        >>> calculate_indegree(graph)
        {1: 0, 2: 1, 3: 2}

        This indicates vertex 1 has an in-degree of 0 (no incoming edges), vertex 2 has
        an in-degree of 1 (one edge directed towards it), and vertex 3 has an in-degree
        of 2 (two edges directed towards it).

    Note:
    ----
        This function assumes the input graph is well-formed and does not check for
        the existence of vertices not listed as keys in the graph dictionary. It is
        expected that every vertex that can be reached from any vertex in the graph
        is also a key in the `graph` dictionary.

    """
    in_degree = {vertex: 0 for vertex in graph}
    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1
    return in_degree


def build_weighted_ugraph(num_vertices: int, edges: list[tuple[list[int], int]]) -> dict[int, set[tuple[int, int]]]:
    """Builds an undirected graph represented as an adjacency list.

    Args:
    ----
        num_vertices (int): The number of vertices in the graph.
        edges (List[Tuple[List[int], int]]): A list of edges where each edge is
        represented as a Tuple ([src, neighbor], weight).

    Returns:
    -------
        Dict[int, Set[Tuple[int, int]]]: A dictionary representing the undirected graph.
        Each key is a vertex, and each value is a list of adjacent vertices and
        respective weight of the edge (neighbor, weight).

    Example:
    -------
        >>> build_weighted_ugraph(3, [([0, 1], 10), ([1, 2], 1), ([2, 0], 5)])
        {0: [(1, 10), (2, 5)], 1: [(0, 10), (2, 1)], 2: [(1, 1), (0, 5)]}

    """
    graph = {vertex: set() for vertex in range(num_vertices)}
    for (src, neighbor), w in edges:
        graph[src].add((neighbor, w))
        graph[neighbor].add((src, w))
    return graph


def transpose_digraph(graph: dict[int, set[int]]) -> dict[int, set[int]]:
    """Transpose a directed graph.

    The transposition of a graph involves reversing the direction of all its edges.
    This function takes a directed graph represented as a dictionary, where each key
    is a vertex and its value is a set of vertices representing the directed edges
    from the key vertex to other vertices. It returns a new graph where the directions
    of all edges have been reversed.

    Args:
    ----
        graph (Dict[int, Set[int]]): The original directed graph to be transposed,
                                     represented as an adjacency list.

    Returns:
    -------
        Dict[int, Set[int]]: The transposed graph, also represented as an adjacency list.

    Example:
    -------
        >>> graph = {1: {2}, 2: {3}, 3: {1, 4}, 4: {5}, 5: {6}, 6: {4, 7}, 7: {8}, 8: set()}
        >>> transpose_graph(graph)
        {1: {3}, 2: {1}, 3: {2}, 4: {3, 6}, 5: {4}, 6: {5}, 7: {6}, 8: {7}, 1: set(), 2: set(), ...}

    """
    transposed = {v: set() for v in graph}
    for vertex in graph:
        for neighbor in graph[vertex]:
            transposed[neighbor].add(vertex)
    return transposed


def plot_graph(graph: dict[int, set[int]]) -> None:
    """Visualizes the given graph

    Each key is a node and its value is a set of connected nodes.

    Args:
    ----
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


def plot_digraph(graph: dict[int, set[tuple[int, int]]]) -> None:
    """Visualizes the given directed graph.

    Args:
    ----
    - graph: A dictionary representing the graph's adjacency list,
             where each key is a node and the value is a set of tuples
             representing directed edges to neighboring nodes.

    """
    DG = nx.DiGraph()

    for node in graph:
        DG.add_node(node)

    for src, edges in graph.items():
        for dest in edges:
            DG.add_edge(src, dest)

    nx.draw(
        DG,
        with_labels=True,
        node_color="lightblue",
        edge_color="k",
        node_size=700,
        font_size=15,
        arrows=True,
        pos=nx.spring_layout(DG, seed=42),  # Positions for all nodes.
    )

    plt.show()


def generate_graph_with_one_odd_cycle(num_nodes: int, cycle_length: int) -> dict[int, set[int]]:
    if not (cycle_length < num_nodes and cycle_length % 2 == 1):
        raise ValueError("Cycle length should be odd and less than total nodes.")

    edges = []
    # Generate a cycle of odd length
    cycle_nodes = random.sample(range(num_nodes), cycle_length)
    for i in range(cycle_length):
        edges.append((cycle_nodes[i], cycle_nodes[(i + 1) % cycle_length]))

    # Connect remaining nodes without forming additional cycles
    remaining_nodes = [node for node in range(num_nodes) if node not in cycle_nodes]
    for i in range(len(remaining_nodes) - 1):
        # Simply connect each remaining node to the next to avoid forming cycles
        edges.append((remaining_nodes[i], remaining_nodes[i + 1]))

    return build_ugraph(num_nodes, edges)


def generate_graph_with_components(
    num_nodes: int, num_components: int, include_cycle: bool = False
) -> dict[int, set[int]]:
    assert num_components <= num_nodes, "Number of components cannot exceed the number of nodes."

    def create_tree(nodes: list[int]) -> list[tuple[int, int]]:
        edges = []
        for i in range(1, len(nodes)):
            edges.append((nodes[i], nodes[random.randint(0, i - 1)]))  # Connect to a random previous node
        return edges

    def add_cycle_to_component(edges: list[tuple[int, int]], nodes: list[int]) -> None:
        cycle_nodes = random.sample(nodes, random.randint(3, min(len(nodes), 5)))  # Create a small cycle, 3-5 nodes
        for i in range(len(cycle_nodes)):
            edges.append((cycle_nodes[i], cycle_nodes[(i + 1) % len(cycle_nodes)]))

    graph: dict[int, set[int]] = {i: set() for i in range(num_nodes)}
    nodes_per_component = [
        num_nodes // num_components + (1 if x < num_nodes % num_components else 0) for x in range(num_components)
    ]

    start = 0
    for i, size in enumerate(nodes_per_component):
        component_nodes = list(range(start, start + size))
        start += size

        # Create a tree component
        component_edges = create_tree(component_nodes)

        # Optionally add a cycle to the first component
        if include_cycle and i == 0:
            add_cycle_to_component(component_edges, component_nodes)

        # Add edges to the graph
        for src, dest in component_edges:
            graph[src].add(dest)
            graph[dest].add(src)  # Assuming undirected graph

    return graph


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

    # graph = build_ugraph(8, [[0, 1], [3, 5], [4, 5], [5, 6], [5, 7]])

    # print(count_components_iterative(graph))
    # print(count_components_recursive(graph))
    # print(largest_component_size(graph))
    # print(largest_component_recursive(graph))
    # print(largest_component_iterative(graph))
    # print("shortest path")
    # print(lenght_shortest_path(graph, 3, 6))
    # print(lenght_shortest_path(graph, 0, 1))
    # print(lenght_shortest_path(graph, 0, 6))
    # plot_graph(graph)

    # plot_graph({0: {1}, 1: {2, 3, 4}, 2: {1, 5}, 3: {1}, 4: {1, 5}, 5: {2, 4}} )
    # plot_graph({0: {1}, 1: {0}, 2: {3}, 3: {2}} )

    # GRAPH_SIMPLE = {0: {1, 2}, 1: {0, 3}, 2: {0}, 3: {1}}  # Simple connected graph
    # GRAPH_CYCLIC = {0: {1}, 1: {2}, 2: {0}}  # Cyclic graph (triangle)
    # GRAPH_COMPLEX = {0: {1}, 1: {2, 3, 4}, 2: {1, 5}, 3: {1}, 4: {1, 5}, 5: {2, 4}}  # More complex graph
    # GRAPH_SMALL_DISCONNECTED = {0: {1}, 1: {0}, 2: {3}, 3: {2}}  # Disconnected graph (two components)
    # GRAPH_BIG_DISCONNECTED = {0: {1}, 1: {0}, 2: {}, 3: {5}, 4: {5}, 5: {3, 4, 6, 7}, 6: {5}, 7: {5}}

    # GRAPHS = [GRAPH_SIMPLE, GRAPH_CYCLIC, GRAPH_COMPLEX, GRAPH_SMALL_DISCONNECTED, GRAPH_BIG_DISCONNECTED]
    # for graph in GRAPHS:
    #     plot_graph(graph)

    wgraph = build_weighted_ugraph(3, [([0, 1], 10), ([1, 2], 1), ([2, 0], 5)])
    print(wgraph)

    # Generate a graph with 50 nodes and one cycle of length 5
    num_nodes = 50
    cycle_length = 5
    graph = generate_graph_with_one_odd_cycle(num_nodes, cycle_length)
    plot_graph(graph)

    # Example Usage
    num_nodes = 20
    num_components = 4
    include_cycle = True

    graph = generate_graph_with_components(num_nodes, num_components, include_cycle)
    plot_graph(graph)
