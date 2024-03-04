import pytest
from leetcode.p_948_bag_of_tokens import bag_of_tokens_score


@pytest.mark.parametrize(
    "tokens, power, expected", [([100], 50, 0), ([200, 100], 150, 1), ([100, 200, 300, 400], 200, 2)]
)
def test_bag_of_tokens_score(tokens, power, expected):
    assert bag_of_tokens_score(tokens, power) == expected
