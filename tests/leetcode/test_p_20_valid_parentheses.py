import pytest

from leetcode.p_20_valid_parentheses import is_valid_parenthesis


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ("()", True),
        ("()[]{}", True),
        ("([()])", True),
        ("([({}{}[()])])", True),
        ("(]", False),
        ("([)]", False),
        ("([)]", False),
        ("([()]", False),
        ("]", False),
    ],
)
def test_is_valid_parenthesis(test_input, expected):
    assert is_valid_parenthesis(test_input) == expected
