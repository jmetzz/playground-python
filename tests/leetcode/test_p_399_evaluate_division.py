import pytest

from leetcode.p_399_evaluate_division import calc_equation_bfs, calc_equation_dfs


@pytest.mark.parametrize("func", [calc_equation_bfs, calc_equation_dfs])
@pytest.mark.parametrize(
    ("equations", "values", "queries", "expected"),
    [
        ([["a", "b"]], [0.5], [["a", "b"], ["b", "a"], ["a", "c"], ["x", "y"]], [0.5, 2.0, -1.0, -1.0]),
        (
            [["a", "b"], ["b", "c"]],
            [2.0, 3.0],
            [["a", "c"], ["b", "a"], ["a", "e"], ["a", "a"], ["x", "x"]],
            [6.0, 0.5, -1.0, 1.0, -1.0],
        ),
        (
            [["a", "b"], ["b", "c"], ["bc", "cd"]],
            [1.5, 2.5, 5.0],
            [["a", "c"], ["c", "b"], ["bc", "cd"], ["cd", "bc"]],
            [3.75, 0.40, 5.0, 0.2],
        ),
    ],
)
def test_evaluate_division(func, equations, values, queries, expected):
    assert func(equations, values, queries) == expected
