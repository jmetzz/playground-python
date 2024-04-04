import pytest
from leetcode.p_1614_maximum_nesting_depth_parentheses import (
    max_depth_cheeting,
    max_depth_counting,
    max_depth_stack,
    max_depth_unrestricted,
)


@pytest.mark.parametrize("func", [max_depth_cheeting, max_depth_counting, max_depth_stack, max_depth_unrestricted])
@pytest.mark.parametrize(
    "input_expression, expected",
    [
        ("", 0),
        ("1", 0),
        ("1+5", 0),
        ("(1+5)", 1),
        ("(1+5)*2", 1),
        ("(1+5)*(2-9)", 1),
        ("((1+5)*(2-9))", 2),
        ("(1+(2*3)+((8)/4))+1", 3),
        ("(1)+((2))+(((3)))", 3),
    ],
)
def test_max_nesting_depth(func, input_expression, expected):
    assert func(input_expression) == expected
