import pytest

from leetcode.p_85_maximal_rectangle import maximal_rectangle


@pytest.mark.parametrize(
    "input_matrix, expected",
    [
        (
            [
                ["1", "0", "1", "0", "0"],
                ["1", "0", "1", "1", "1"],
                ["1", "1", "1", "1", "1"],
                ["1", "0", "0", "1", "0"],
            ],
            6,
        ),
        ([["0"]], 0),
        ([["1"]], 1),
    ],
)
def test_maximal_rectangle(input_matrix, expected):
    assert maximal_rectangle(input_matrix) == expected
