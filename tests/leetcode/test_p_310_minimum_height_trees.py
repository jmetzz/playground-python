import pytest

from leetcode.p_310_minimum_height_trees import (
    find_min_height_trees_dfs,
    find_min_height_trees_removing_leaves,
)


@pytest.mark.parametrize("func", [find_min_height_trees_dfs, find_min_height_trees_removing_leaves])
@pytest.mark.parametrize(
    "input_n, input_graph, expected",
    [
        (4, [[1, 0], [1, 2], [1, 3]], [1]),
        (6, [[3, 0], [3, 1], [3, 2], [3, 4], [5, 4]], [3, 4]),
        (10, [[0, 3], [1, 3], [2, 3], [4, 3], [5, 3], [4, 6], [4, 7], [5, 8], [5, 9]], [3]),
    ],
)
def test_find_min_height_trees(func, input_n, input_graph, expected):
    assert func(input_n, input_graph) == expected
