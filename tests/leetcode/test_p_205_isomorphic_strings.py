import pytest
from leetcode.p_205_isomorphic_strings import is_isomorphic_magic, is_isomorphic_one_dict, is_isomorphic_two_dicts

test_cases = [
    ("egg", "add", True),
    ("paper", "title", True),
    ("foo", "bar", False),
    ("egk", "add", False),
    ("bbbaaaba", "aaabbbba", False),
]


@pytest.mark.parametrize("func", [is_isomorphic_magic, is_isomorphic_one_dict, is_isomorphic_two_dicts])
@pytest.mark.parametrize("input_s, input_t, expected", test_cases)
def test_is_isomorph(func, input_s, input_t, expected):
    assert func(input_s, input_t) == expected
