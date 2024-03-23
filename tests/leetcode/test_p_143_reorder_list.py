import pytest
from data_structures.lists import SingleLinkNode as ListNode
from leetcode.p_143_reorder_list import (
    reorder_list,
    reorder_list_even_and_reversed_odd_indices,
    reorder_list_fast_slow_pointers,
)

test_cases = [
    ([1], [1]),
    ([1, 2], [1, 2]),
    ([1, 2, 3], [1, 3, 2]),
    ([1, 2, 3, 4], [1, 4, 2, 3]),
    ([1, 2, 3, 4, 5], [1, 5, 2, 4, 3]),
]


@pytest.mark.parametrize("func", [reorder_list, reorder_list_fast_slow_pointers])
@pytest.mark.parametrize("input_arr, expected_arr", test_cases)
def test_reorder_list(func, input_arr, expected_arr):
    list_under_test = ListNode.deserialize(input_arr)
    func(list_under_test)
    assert list_under_test.as_array() == expected_arr


test_cases_reversed = [
    ([1], [1]),
    ([1, 2], [1, 2]),
    ([1, 2, 3], [1, 3, 2]),
    ([1, 2, 3, 4], [1, 4, 3, 2]),
    ([1, 2, 3, 4, 5], [1, 5, 4, 2, 3]),
    ([1, 2, 3, 4, 5, 6], [1, 6, 5, 2, 3, 4]),
    ([1, 2, 3, 4, 5, 6, 7], [1, 7, 6, 2, 3, 5, 4]),
]


@pytest.mark.parametrize("input_arr, expected_arr", test_cases_reversed)
def test_reorder_list_even_and_reversed_odd_indices(input_arr, expected_arr):
    list_under_test = ListNode.deserialize(input_arr)
    reorder_list_even_and_reversed_odd_indices(list_under_test)
    assert list_under_test.as_array() == expected_arr
