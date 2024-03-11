from collections import Counter

import pytest
from leetcode.p_791_custom_sort_string import custom_sort_string_1, custom_sort_string_2, custom_sort_string_3

# List of sorting functions to be tested
sort_functions = [
    custom_sort_string_1,
    custom_sort_string_2,
    custom_sort_string_3,
]

# Test cases as tuples: (order, s, expected_result)
test_cases__all_chars_exist_in_order_str = [
    ("abc", "cbad", "abcd"),
    ("xyz", "yxz", "xyz"),
    ("cba", "abcd", "cbad"),
]

# Test cases as tuples: (order, s, expected_result)
test_cases__some_chars_not_in_order_str = [
    ("abc", "cbad", "abc"),
    ("xyz", "yxz", "xyz"),
    ("cba", "abcd", "cba"),
    ("cba", "abcdzzz", "cba"),
    ("cba", "abcdzzza", "cbaa"),
]


@pytest.mark.parametrize("sort_function", sort_functions)
@pytest.mark.parametrize("order, s, expected", test_cases__all_chars_exist_in_order_str)
def test_custom_sort_string__only_chars_in_the_order_obj(sort_function, order, s, expected):
    actual = sort_function(order, s)
    assert actual == expected, f"Expected {expected}, got {actual} instead."


@pytest.mark.parametrize("sort_function", sort_functions)
@pytest.mark.parametrize("order, s, expected", test_cases__some_chars_not_in_order_str)
def test_custom_sort_string__with_chars_not_in_the_order_obj(sort_function, order, s, expected):
    actual = sort_function(order, s)

    assert Counter(actual) == Counter(s)

    # Chars that are not in the order can appear in any order
    chars_to_remove = str(set(s) - set(order))

    # Create a translation table that maps characters to None if they should be removed
    trans_table = str.maketrans("", "", chars_to_remove)
    # Use the translation table to remove the specified characters
    actual = actual.translate(trans_table)
    assert actual == expected, f"Relevant sorted portion: Expected {expected}, got {actual} instead."
