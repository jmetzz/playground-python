import pytest

from leetcode.p_997_find_the_town_judge import FindTownJudge


@pytest.mark.parametrize(
    "input_n, input_trust, expected",
    [
        (2, [[1, 2]], 2),
        (3, [[1, 3], [2, 3]], 3),
        (3, [[1, 3], [2, 3], [3, 1]], -1),
        (1, [], 1),
    ],
)
def test_find_town_judge(input_n, input_trust, expected):
    assert FindTownJudge().solve(input_n, input_trust) == expected


@pytest.mark.parametrize(
    "input_n, input_trust, expected",
    [
        (2, [[1, 2]], 2),
        (3, [[1, 3], [2, 3]], 3),
        (3, [[1, 3], [2, 3], [3, 1]], -1),
        (1, [], 1),
    ],
)
def test_find_town_judge_graph(input_n, input_trust, expected):
    assert FindTownJudge().solve_graph(input_n, input_trust) == expected
