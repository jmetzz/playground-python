import pytest

from leetcode.p_2390_removing_stars_from_a_string import remove_stars


@pytest.mark.parametrize("test_input, expected", [("leet**cod*e", "lecoe"), ("erase*****", "")])
def test_remove_stars(test_input, expected):
    assert remove_stars(test_input) == expected
