import pytest
from leetcode.p_77_combinations import combine


@pytest.mark.parametrize("n, k, expected", [(4, 2, [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]), (1, 1, [[1]])])
def test_combine(n, k, expected):
    actual = combine(n, k)
    assert actual == expected
