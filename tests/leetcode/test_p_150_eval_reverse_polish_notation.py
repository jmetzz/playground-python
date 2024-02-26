import pytest
from leetcode.p_150_eval_reverse_polish_notation import eval_reverse_polish_notation


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (["2", "1", "+", "3", "*"], 9),
        (["4", "13", "5", "/", "+"], 6),
        (["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"], 22),
    ],
)
def test_eval_rvn(test_input, expected):
    assert eval_reverse_polish_notation(test_input) == expected
