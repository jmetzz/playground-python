import pytest
from data_structures.graphs import (
    count_components_iterative,
    count_components_recursive,
    has_path_bfs_iterative,
    has_path_bfs_recursive,
    is_bipartite_bfs,
    is_bipartite_dfs,
    kosaraju,
    largest_component_iterative,
    largest_component_recursive,
    largest_component_size,
    shortest_path_lenght,
    traversal_bfs,
    traversal_dfs_iterative,
    traversal_dfs_recursive,
)

# Graph definitions
GRAPH_SIMPLE_CONNECTED = {0: set([1, 2]), 1: set([0, 3]), 2: set([0]), 3: set([1])}
GRAPH_CYCLIC_CONNECTED = {0: set([1]), 1: set([2]), 2: set([0])}  # triangle
GRAPH_COMPLEX_CONNECTED = {0: set([1]), 1: set([2, 3, 4]), 2: set([1, 5]), 3: set([1]), 4: set([1, 5]), 5: set([2, 4])}
GRAPH_SMALL_DISCONNECTED = {0: set([1]), 1: set([0]), 2: set([3]), 3: set([2])}  # two components
GRAPH_BIG_DISCONNECTED = {
    0: set([1]),
    1: set([0]),
    2: set([]),
    3: set([5]),
    4: set([5]),
    5: set([3, 4, 6, 7]),
    6: set([5]),
    7: set([5]),
}  # 2 components

GRAPHS = [
    GRAPH_SIMPLE_CONNECTED,
    GRAPH_CYCLIC_CONNECTED,
    GRAPH_COMPLEX_CONNECTED,
    GRAPH_SMALL_DISCONNECTED,
    GRAPH_BIG_DISCONNECTED,
]


GRAPH_BIPARTITE = {
    # In this graph, you can color nodes 0, 2, 3, 5, 6, 8 with one color, and 1, 4, 7 with another,
    # ensuring no two adjacent nodes share the same color.
    0: {1, 3},
    1: {0, 2, 4},
    2: {1, 5},
    3: {0, 4, 6},
    4: {1, 3, 5, 7},
    5: {2, 4, 8},
    6: {3, 7},
    7: {4, 6, 8},
    8: {5, 7},
    9: set(),  # Adding a disconnected node to make up 10 nodes
}


GRAPH_NON_BIPARTITE = {
    # A non-bipartite graph contains at least one set of nodes where at least one node is adjacent to
    # another node in the same set. Introducing a cycle of odd length in a graph ensures it's non-bipartite.
    0: {1, 3},
    1: {0, 2},
    2: {1, 5},
    3: {0, 4, 6},
    4: {3, 6},  # This creates an odd-length cycle with 3-4-6
    5: {2, 8},
    6: {3, 7},
    7: {6, 8, 9},
    8: {5, 7},
    9: {},
}

test_cases_for_has_path = [
    ({}, 1, 2, False),  # Empty graph
    ({1: {2}, 2: {1}}, 1, 2, True),  # Cycle
    ({1: {2}, 2: {3}, 3: {4}, 4: set()}, 1, 1, True),  # Source equals target
    ({1: {2, 3}, 2: {4}, 3: {4}, 4: set()}, 1, 4, True),  # Basic case with path
    ({1: {2}, 2: {3}, 3: {4}, 4: set()}, 1, 3, True),  # Path exists
    ({1: {2, 3}, 2: {5}, 3: {5}, 5: set()}, 1, 4, False),  # No path
    ({1: {2}, 2: {3}, 3: {4}, 4: set()}, 4, 1, False),  # Reverse path does not exist
]


@pytest.mark.parametrize("graph, source, target, expected", test_cases_for_has_path)
def test_has_path_bfs_recursive_basec_cases(graph, source, target, expected):
    assert has_path_bfs_recursive(graph, source, target) == expected


@pytest.mark.parametrize("graph, source, target, expected", test_cases_for_has_path)
def test_has_path_bfs_iterative_basec_cases(graph, source, target, expected):
    assert has_path_bfs_iterative(graph, source, target) == expected


@pytest.mark.parametrize(
    "graph,source,destination,expected",
    [
        (GRAPH_SIMPLE_CONNECTED, 0, 3, True),
        (GRAPH_CYCLIC_CONNECTED, 0, 2, True),
        (GRAPH_SMALL_DISCONNECTED, 0, 1, True),
        (GRAPH_BIG_DISCONNECTED, 0, 5, False),
        (GRAPH_BIG_DISCONNECTED, 0, 1, True),
        (GRAPH_BIG_DISCONNECTED, 3, 4, True),
        (GRAPH_BIG_DISCONNECTED, 2, 0, False),
    ],
)
def test_has_path_bfs_recursive(graph, source, destination, expected):
    assert has_path_bfs_recursive(graph, source, destination) == expected


@pytest.mark.parametrize(
    "graph,source,destination,expected",
    [
        (GRAPH_SIMPLE_CONNECTED, 0, 3, True),
        (GRAPH_CYCLIC_CONNECTED, 0, 2, True),
        (GRAPH_SMALL_DISCONNECTED, 0, 1, True),
        (GRAPH_BIG_DISCONNECTED, 0, 5, False),
        (GRAPH_BIG_DISCONNECTED, 0, 1, True),
        (GRAPH_BIG_DISCONNECTED, 3, 4, True),
        (GRAPH_BIG_DISCONNECTED, 2, 0, False),
    ],
)
def test_has_path_bfs_iterative(graph, source, destination, expected):
    assert has_path_bfs_iterative(graph, source, destination) == expected


@pytest.mark.parametrize(
    "graph,source,expected",
    [
        (GRAPH_SIMPLE_CONNECTED, 0, [0, 2, 1, 3]),
        (GRAPH_CYCLIC_CONNECTED, 0, [0, 2, 1]),
        (GRAPH_COMPLEX_CONNECTED, 0, [0, 1, 4, 5, 2, 3]),
        (GRAPH_SMALL_DISCONNECTED, 0, [0, 1]),
    ],
)
def test_dfs_graph_traversal_iterative(graph, source, expected):
    assert set(traversal_dfs_iterative(graph, source)) == set(expected)


@pytest.mark.parametrize(
    "graph,source,expected",
    [
        (GRAPH_SIMPLE_CONNECTED, 0, [0, 1, 3, 2]),
        (GRAPH_CYCLIC_CONNECTED, 0, [0, 1, 2]),
        (GRAPH_COMPLEX_CONNECTED, 0, [0, 1, 2, 5, 4, 3]),
        (GRAPH_SMALL_DISCONNECTED, 0, [0, 1]),
    ],
)
def test_dfs_graph_traversal_recursive(graph, source, expected):
    assert traversal_dfs_recursive(graph, source) == expected


@pytest.mark.parametrize(
    "graph,source,expected",
    [
        (GRAPH_SIMPLE_CONNECTED, 0, [0, 1, 2, 3]),
        (GRAPH_CYCLIC_CONNECTED, 0, [0, 1, 2]),
        (GRAPH_COMPLEX_CONNECTED, 0, [0, 1, 2, 3, 4, 5]),
        (GRAPH_SMALL_DISCONNECTED, 0, [0, 1]),
    ],
)
def test_bfs_graph_traversal(graph, source, expected):
    assert traversal_bfs(graph, source) == expected


@pytest.mark.parametrize(
    "graph,expected",
    [
        (GRAPH_SIMPLE_CONNECTED, 1),
        (GRAPH_CYCLIC_CONNECTED, 1),
        (GRAPH_COMPLEX_CONNECTED, 1),
        (GRAPH_SMALL_DISCONNECTED, 2),
        (GRAPH_BIG_DISCONNECTED, 3),
    ],
)
def test_count_components_iterative(graph, expected):
    assert count_components_iterative(graph) == expected


@pytest.mark.parametrize(
    "graph,expected",
    [
        (GRAPH_SIMPLE_CONNECTED, 1),
        (GRAPH_CYCLIC_CONNECTED, 1),
        (GRAPH_COMPLEX_CONNECTED, 1),
        (GRAPH_SMALL_DISCONNECTED, 2),
        (GRAPH_BIG_DISCONNECTED, 3),
    ],
)
def test_count_components_recursive(graph, expected):
    assert count_components_recursive(graph) == expected


@pytest.mark.parametrize(
    "graph,expected",
    [
        (GRAPH_SIMPLE_CONNECTED, 4),
        (GRAPH_CYCLIC_CONNECTED, 3),
        (GRAPH_COMPLEX_CONNECTED, 6),
        (GRAPH_SMALL_DISCONNECTED, 2),
        (GRAPH_BIG_DISCONNECTED, 5),
    ],
)
def test_largest_component_size(graph, expected):
    assert largest_component_size(graph) == expected


@pytest.mark.parametrize(
    "graph,expected",
    [
        (GRAPH_SIMPLE_CONNECTED, set([0, 1, 2, 3])),
        (GRAPH_CYCLIC_CONNECTED, set([0, 1, 2])),
        (GRAPH_COMPLEX_CONNECTED, set([0, 1, 2, 3, 4, 5])),
        (GRAPH_BIG_DISCONNECTED, set([3, 4, 5, 6, 7])),
    ],
)
def test_largest_component_recursive(graph, expected):
    assert set(largest_component_recursive(graph)) == expected


@pytest.mark.parametrize(
    "graph,expected",
    [
        (GRAPH_SIMPLE_CONNECTED, set([0, 1, 2, 3])),
        (GRAPH_CYCLIC_CONNECTED, set([0, 1, 2])),
        (GRAPH_COMPLEX_CONNECTED, set([0, 1, 2, 3, 4, 5])),
        (GRAPH_BIG_DISCONNECTED, set([3, 4, 5, 6, 7])),
    ],
)
def test_largest_component_iterative(graph, expected):
    assert set(largest_component_iterative(graph)) == expected


@pytest.mark.parametrize(
    "graph, source, destination, expected",
    [
        (GRAPH_SIMPLE_CONNECTED, 1, 3, 2),
        (GRAPH_SIMPLE_CONNECTED, 1, 2, 3),
        (GRAPH_CYCLIC_CONNECTED, 1, 2, 2),
        (GRAPH_COMPLEX_CONNECTED, 0, 5, 4),
        (GRAPH_COMPLEX_CONNECTED, 0, 1, 2),
        (GRAPH_COMPLEX_CONNECTED, 0, 3, 3),
        (GRAPH_BIG_DISCONNECTED, 0, 0, 1),
        (GRAPH_BIG_DISCONNECTED, 0, 1, 2),
        (GRAPH_BIG_DISCONNECTED, 0, 2, 0),
        (GRAPH_BIG_DISCONNECTED, 0, 4, 0),
        (GRAPH_BIG_DISCONNECTED, 3, 4, 3),
        (GRAPH_BIG_DISCONNECTED, 5, 7, 2),
    ],
)
def test_lenght_shortest_path(graph, source, destination, expected):
    assert shortest_path_lenght(graph, source, destination) == expected


@pytest.mark.parametrize("func", [is_bipartite_dfs, is_bipartite_bfs])
@pytest.mark.parametrize(
    "input_graph, expected",
    [
        (GRAPH_BIPARTITE, True),
        (GRAPH_NON_BIPARTITE, False),
    ],
)
def test_is_bipartite(func, input_graph, expected):
    assert func(input_graph) == expected


@pytest.mark.parametrize(
    "graph, expected",
    [
        ({1: {2}, 2: {3}, 3: {1}}, [[1, 3, 2]]),
        ({1: {2}, 2: {1}, 3: {4}, 4: {3}}, [[1, 2], [3, 4]]),
        ({1: {2}, 2: {3, 4}, 3: {1}, 4: {5}, 5: {6}, 6: {4}, 7: {6, 8}, 8: {}}, [[1, 3, 2], [4, 6, 5], [7], [8]]),
        ({1: set()}, [[1]]),
        ({1: {2}, 2: {3}, 3: {4}, 4: {}}, [[1], [2], [3], [4]]),
    ],
)
def test_kosaraju(graph, expected):
    result = kosaraju(graph)
    # Convert both expected and result to sets of frozensets for comparison to avoid order issues
    expected_set = {frozenset(component) for component in expected}
    result_set = {frozenset(component) for component in result}
    assert expected_set == result_set
