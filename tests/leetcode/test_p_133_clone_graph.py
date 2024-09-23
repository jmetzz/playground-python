import pytest

from leetcode.p_133_clone_graph import (
    are_graphs_equal,
    build_graph,
    clone_graph_bfs,
    clone_graph_dfs,
    clone_graph_dfs_explicit_stack,
)


@pytest.mark.parametrize("func", [clone_graph_dfs, clone_graph_dfs_explicit_stack, clone_graph_bfs])
@pytest.mark.parametrize(
    "adj_list",
    [
        ([]),
        ([[]]),
        ([[2, 4], [1, 3], [2, 4], [1, 3]]),
    ],
)
def test_clone_graph(func, adj_list):
    graph = build_graph(adj_list)
    assert are_graphs_equal(graph, func(graph), set())
