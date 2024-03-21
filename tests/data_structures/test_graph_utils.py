import pytest
from data_structures.graph_utils import build_digraph, build_ugraph


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
