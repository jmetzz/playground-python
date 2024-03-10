import pytest
from algorithms.sorting import (
    counting_sort,
    merge_sort,
    quick_sort_hoare,
    quick_sort_lomuto,
    quick_sort_median_of_three,
    quick_sort_random_pivot,
    quick_sort_three_way,
)
from hypothesis import given, settings, strategies

# List of sorting functions to be tested
sorting_functions = [
    (merge_sort, "merge_sort"),
    (quick_sort_lomuto, "quick_sort_lomuto"),
    (quick_sort_hoare, "quick_sort_hoare"),
    (quick_sort_random_pivot, "quick_sort_random_pivot"),
    (quick_sort_median_of_three, "quick_sort_median_of_three"),
    (quick_sort_three_way, "quick_sort_three_way"),
]

# Test cases as a list of tuples (input_list, expected_sorted_list)
test_cases = [
    ([], []),
    ([1], [1]),
    ([2, 1], [1, 2]),
    ([3, 1, 2], [1, 2, 3]),
    ([5, -1, 6, 2], [-1, 2, 5, 6]),
    ([4, 3, 2, 1], [1, 2, 3, 4]),
    ([10, -2, 5, -1, 7, 6, -5, 3], [-5, -2, -1, 3, 5, 6, 7, 10]),
]


test_cases__positives_only = [
    ([], []),
    ([1], [1]),
    ([2, 1], [1, 2]),
    ([3, 1, 2], [1, 2, 3]),
    ([5, 1, 6, 2], [1, 2, 5, 6]),
    ([4, 3, 2, 1], [1, 2, 3, 4]),
    ([10, 2, 5, 1, 7, 6, 5, 3], [1, 2, 3, 5, 5, 6, 7, 10]),
]


# Parametrize sorting functions
@pytest.mark.parametrize("sorting_function, funcion_name", sorting_functions)
# Parametrize test cases
@pytest.mark.parametrize("input_list, expected", test_cases)
def test_sorting_methods(sorting_function, funcion_name, input_list, expected):
    assert sorting_function(input_list.copy()) == expected, f"Failed on {funcion_name} with input {input_list}"


# Strategy for generating lists of non-negative integers
non_negative_int_list = strategies.lists(strategies.integers(min_value=0, max_value=100), min_size=0, max_size=100)


@given(non_negative_int_list)
@settings(max_examples=20)
def test_counting_sort(input_list):
    actual = counting_sort(input_list.copy())
    assert actual == sorted(input_list)
    assert len(actual) == len(input_list), "Sorted array length differs from input array length."
