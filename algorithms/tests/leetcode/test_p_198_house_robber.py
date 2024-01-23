import pytest

from leetcode.p_198_house_robber import HouseRobber


@pytest.mark.parametrize("test_input, expected", [([1, 2, 3, 1], 4), ([2, 7, 9, 3, 1], 12)])
def test_rob(test_input, expected):
    assert HouseRobber().rob(test_input) == expected
