import pytest

from leetcode.p_198_house_robber import house_robber


@pytest.mark.parametrize("test_input, expected", [([1, 2, 3, 1], 4), ([2, 7, 9, 3, 1], 12)])
def test_house_robber(test_input, expected):
    assert house_robber(test_input) == expected
