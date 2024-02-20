import pytest
from leetcode.p_387_first_unique_character_in_string import FirstUniqChar


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ("leetcode", 0),
        ("loveleetcode", 2),
        ("aabb", -1),
        ("aabbc", 4),
        ("abracadabra", 4),
        ("abracadabrac", 6),
    ],
)
def test_first_unique_character_in_string(test_input, expected):
    assert FirstUniqChar().solve(test_input) == expected
