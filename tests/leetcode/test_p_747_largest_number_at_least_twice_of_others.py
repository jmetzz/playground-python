import pytest
from leetcode.p_747_largest_number_at_least_twice_of_others import dominant_index, dominant_index_heap


@pytest.mark.parametrize("input_array, expected", [([3, 0], 0), ([0, 3], 1), ([3, 6, 1, 0], 1), ([1, 2, 3, 4], -1)])
def test_dominant_index(input_array, expected):
    assert dominant_index(input_array) == expected


@pytest.mark.parametrize("input_array, expected", [([3, 0], 0), ([0, 3], 1), ([3, 6, 1, 0], 1), ([1, 2, 3, 4], -1)])
def test_dominant_index_heap(input_array, expected):
    assert dominant_index_heap(input_array) == expected
