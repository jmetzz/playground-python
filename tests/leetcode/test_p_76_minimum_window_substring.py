import pytest
from leetcode.p_76_minimum_window_substring import min_window_solve


@pytest.mark.parametrize(
    "test_input_s, test_input_t, expected",
    [
        ("ADOBECODEBANC", "ABC", "BANC"),
        ("a", "a", "a"),
        ("a", "aa", ""),
    ],
)
def test_k_inverse_pairs_array_memo(test_input_s, test_input_t, expected):
    assert min_window_solve(test_input_s, test_input_t) == expected
