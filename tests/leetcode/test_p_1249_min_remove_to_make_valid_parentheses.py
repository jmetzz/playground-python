import pytest
from leetcode.p_1249_min_remove_to_make_valid_parentheses import min_remove_to_make_valid


@pytest.mark.parametrize(
    "input_text, expected", [("lee(t(c)o)de)", "lee(t(c)o)de"), ("a)b(c)d", "ab(c)d"), ("))((", "")]
)
def test_min_remove_to_make_valid(input_text, expected):
    assert min_remove_to_make_valid(input_text) == expected
