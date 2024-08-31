import pytest

from leetcode.p_71_simplify_path import simplify_path, simplify_path_match_case


@pytest.mark.parametrize("func", [simplify_path, simplify_path_match_case])  # noqa: F821
@pytest.mark.parametrize(
    "input_path, expected",
    [
        ("/", "/"),
        ("/.", "/"),
        ("/./", "/"),
        ("/../", "/"),
        ("/home", "/home"),
        ("/home/", "/home"),
        ("/home//foo/", "/home/foo"),
        ("/home/user/Documents/../Pictures", "/home/user/Pictures"),
        ("/.../a/../b/c/../d/./", "/.../b/d"),
    ],
)
def test_function_name(func, input_path, expected):
    assert func(input_path) == expected
