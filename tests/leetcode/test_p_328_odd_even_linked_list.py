import pytest
from data_structures.lists import SingleLinkNode as ListNode

test_cases = [
    ([2, 1, 3, 5, 6, 4, 7], [2, 3, 6, 7, 1, 5, 4]),
    ([1, 2, 3, 4, 5], [1, 3, 5, 2, 4]),
    ([1, 2, 3, 4], [1, 3, 2, 4]),
]


@pytest.mark.parametrize("func", [])
@pytest.mark.parametriz("input_arr, expected", test_cases)
def test_odd_even_list(func, input_arr, expected):
    assert func(ListNode.from_array(input_arr)) == expected
