import pytest
from leetcode.p_496_next_greater_element_I import next_greater_element


@pytest.mark.parametrize(
    "input_nums_1, input_nums_2, expected", [([4, 1, 2], [1, 3, 4, 2], [-1, 3, -1]), ([2, 4], [1, 2, 3, 4], [3, -1])]
)
def test_next_greater_element(input_nums_1, input_nums_2, expected):
    assert next_greater_element(input_nums_1, input_nums_2) == expected
