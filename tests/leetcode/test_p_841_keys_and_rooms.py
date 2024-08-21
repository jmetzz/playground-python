import pytest

from leetcode.p_841_keys_and_rooms import can_visit_all_rooms_bfs


@pytest.mark.parametrize(
    "input_rooms, expected",
    [
        ([[1], [2], [3], []], True),
        ([[1, 3], [3, 0, 1], [2], [0]], False),
        ([[6, 7, 8], [5, 4, 9], [], [8], [4], [], [1, 9, 2, 3], [7], [6, 5], [2, 3, 1]], True),
    ],
)
def test_(input_rooms, expected):
    assert can_visit_all_rooms_bfs(input_rooms) == expected
