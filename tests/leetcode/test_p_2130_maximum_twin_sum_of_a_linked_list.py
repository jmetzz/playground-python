import pytest

from data_structures.lists import SingleLinkNode as ListNode
from leetcode.p_2130_maximum_twin_sum_of_a_linked_list import (
    pair_sum_fast_slow,
    pair_sum_reverse_first_half,
    pair_sum_reverse_second_half,
)

test_cases = [
    ([5, 4, 2, 1], 6),
    ([4, 2, 2, 3], 7),
    ([1, 100000], 100001),
]


@pytest.mark.parametrize("func", [pair_sum_fast_slow, pair_sum_reverse_first_half, pair_sum_reverse_second_half])
@pytest.mark.parametrize("input_arr, expected", test_cases)
def test_twin_sum(func, input_arr, expected):
    assert func(ListNode.deserialize(input_arr)) == expected
