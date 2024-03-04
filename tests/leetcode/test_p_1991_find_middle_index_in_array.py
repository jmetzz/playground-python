import pytest
from leetcode.p_1991_find_middle_index_in_array import find_middle_index


@pytest.mark.parametrize("input_array, expected", [([2, 3, -1, 8, 4], 3), ([1, -1, 4], 2), ([2, 5], -1)])
def test_find_middle_index(input_array, expected):
    assert find_middle_index(input_array) == expected
