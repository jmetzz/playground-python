import pytest
from leetcode.p_344_reverse_string import reverse_inplace


@pytest.mark.parametrize(
    "input_list, expected",
    [
        (["h", "e", "l", "l", "o"], ["o", "l", "l", "e", "h"]),
        (["H", "a", "n", "n", "a", "h"], ["h", "a", "n", "n", "a", "H"]),
        (["h", "e", "l", "l", "o"], ["o", "l", "l", "e", "h"]),
        (list(range(10)), list(reversed(range(10)))),
        (list(range(7)), list(reversed(range(7)))),
    ],
)
def test_reverse_inplace(input_list, expected):
    reverse_inplace(input_list)
    assert input_list == expected
