import pytest
from data_structures.graphs import (
    bfs_graph_traversal,
    build_digraph,
    build_ugraph,
    count_components_iterative,
    count_components_recursive,
    dfs_graph_traversal_iterative,
    dfs_graph_traversal_recursive,
    has_path_bfs_iterative,
    has_path_bfs_recursive,
    largest_component_iterative,
    largest_component_recursive,
    largest_component_size,
    lenght_shortest_path,
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


test_cases_for_has_path = [
    ({}, 1, 2, False),  # Empty graph
    ({1: {2}, 2: {1}}, 1, 2, True),  # Cycle
    ({1: {2}, 2: {3}, 3: {4}, 4: set()}, 1, 1, True),  # Source equals target
    ({1: {2, 3}, 2: {4}, 3: {4}, 4: set()}, 1, 4, True),  # Basic case with path
    ({1: {2}, 2: {3}, 3: {4}, 4: set()}, 1, 3, True),  # Path exists
    ({1: {2, 3}, 2: {5}, 3: {5}, 5: set()}, 1, 4, False),  # No path
    ({1: {2}, 2: {3}, 3: {4}, 4: set()}, 4, 1, False),  # Reverse path does not exist
]


@pytest.mark.parametrize(
    "num_vertices, edges, expected",
    [
        (3, [[0, 1], [1, 2], [2, 0]], {0: set([1, 2]), 1: set([0, 2]), 2: set([0, 1])}),
        (4, [[0, 1], [2, 3]], {0: set([1]), 1: set([0]), 2: set([3]), 3: set([2])}),
        (1, [], {0: set()}),
        (3, [], {0: set(), 1: set(), 2: set()}),
    ],
)
def test_build_ugraph(num_vertices, edges, expected):
    assert build_ugraph(num_vertices, edges) == expected


@pytest.mark.parametrize(
    "num_vertices, edges, calc_indegree, expected_graph, expected_indegree",
    [
        (3, [[0, 1], [1, 2]], False, {0: set([1]), 1: set([2]), 2: set()}, None),
        (3, [[0, 1], [1, 2]], True, {0: set([1]), 1: set([2]), 2: set()}, {0: 0, 1: 1, 2: 1}),
        (
            4,
            [[0, 1], [2, 3], [3, 0]],
            True,
            {0: set([1]), 1: set(), 2: set([3]), 3: set([0])},
            {0: 1, 1: 1, 2: 0, 3: 1},
        ),
        (1, [], False, {0: set()}, None),
        (1, [], True, {0: set()}, {0: 0}),
    ],
)
def test_build_digraph(num_vertices, edges, calc_indegree, expected_graph, expected_indegree):
    graph, indegree = build_digraph(num_vertices, edges, calc_indegree)
    assert graph == expected_graph
    assert indegree == expected_indegree


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
    assert set(dfs_graph_traversal_iterative(graph, source)) == set(expected)


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
    assert dfs_graph_traversal_recursive(graph, source) == expected


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
    assert bfs_graph_traversal(graph, source) == expected


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
    assert lenght_shortest_path(graph, source, destination) == expected
