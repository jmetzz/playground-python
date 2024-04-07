import pytest
from leetcode.p_678_valid_parenthesis_string import check_valid_string_recursive, check_valid_string_two_stacks


@pytest.mark.parametrize("func", [check_valid_string_recursive, check_valid_string_two_stacks])
@pytest.mark.parametrize(
    "input_string, expected",
    [
        ("()", True),
        ("(*)", True),
        ("(*))", True),
        ("(*()))*(", False),
        ("(((()*))*(()(()*((((*)(((()(*())(((()*((())*))))(()(()())())((*()))((((()(*(())((()(()))((()()()*()", False),
    ],
)
def test_check_valid_string(func, input_string, expected):
    assert func(input_string) == expected
