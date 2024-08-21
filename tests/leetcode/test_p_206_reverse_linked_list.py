import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from data_structures.lists import SingleLinkNode as ListNode
from leetcode.p_206_reverse_linked_list import reverse_list_with_pointers, reverse_list_with_stack


@pytest.mark.parametrize("input_arr, expected", [(None, None), ([1], [1]), ([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])])
@pytest.mark.parametrize("func", [reverse_list_with_stack, reverse_list_with_pointers])
def test_reverse_linked_list(func, input_arr, expected):
    actual = func(ListNode.deserialize(input_arr))
    if expected is None:
        assert actual is None
    else:
        assert ListNode.as_array(actual) == expected


@pytest.fixture(params=[reverse_list_with_stack, reverse_list_with_pointers])
def function_under_test(request):
    """Define a fixture that provides the functions to be tested"""
    return request.param


@given(lst=st.lists(st.integers()))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_reverse_list_property_based(lst, function_under_test):
    head = ListNode.deserialize(lst)
    actual = function_under_test(head)
    assert ListNode.as_array(actual) == lst[::-1]
