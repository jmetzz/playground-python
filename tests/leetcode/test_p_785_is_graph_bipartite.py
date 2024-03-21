import pytest
from leetcode.p_785_is_graph_bipartite import is_bipartite_bfs, is_bipartite_dfs

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
