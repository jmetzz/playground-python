import pytest
from leetcode.p_1910_remove_all_occurrences_substring import (
    remove_ccurrences_bool_flag,
    remove_ccurrences_bool_flag_2,
    remove_ccurrences_using_builtins,
)


@pytest.mark.parametrize(
    "func", [remove_ccurrences_using_builtins, remove_ccurrences_bool_flag, remove_ccurrences_bool_flag_2]
)
@pytest.mark.parametrize("text, pattern, expected", [("daabcbaabcbc", "abc", "dab"), ("axxxxyyyyb", "xy", "ab")])
def test_remove_ccurrences(func, text, pattern, expected):
    assert func(text, pattern) == expected
