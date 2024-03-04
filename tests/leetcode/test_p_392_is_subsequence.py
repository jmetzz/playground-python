import pytest
from leetcode.p_392_is_subsequence import is_subsequence, is_subsequence_2


@pytest.mark.parametrize(
    "s, t, expected",
    [
        ("abc", "ahbgdc", True),
        ("axc", "ahbgdc", False),
        ("acb", "ahbgdc", False),
        ("aaaaaa", "bbaaaa", False),
    ],
)
def test_is_subsequence_2(s, t, expected):
    assert is_subsequence_2(s, t) == expected


@pytest.mark.parametrize(
    "s, t, expected",
    [
        ("abc", "ahbgdc", True),
        ("axc", "ahbgdc", False),
        ("acb", "ahbgdc", False),
        ("aaaaaa", "bbaaaa", False),
    ],
)
def test_is_subsequence(s, t, expected):
    assert is_subsequence(s, t) == expected
