import pytest

from algorithms.arrays import reverse, reverse_from, reverse_inplace


@pytest.mark.parametrize(
    "input_list, expected_output",
    [
        ([], []),  # Empty list
        ([1], [1]),  # Single element
        ([1, 2, 3], [3, 2, 1]),  # Multiple elements
        ([1, 2, 3, 4], [4, 3, 2, 1]),  # Multiple elements
        ([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]),  # Multiple elements
    ],
)
def test_reverse(input_list, expected_output):
    assert reverse(input_list) == expected_output


@pytest.mark.parametrize(
    "input_list, start, expected_output",
    [
        ([], 0, []),  # Empty list
        ([1, 2, 3, 4, 5], 0, [5, 4, 3, 2, 1]),  # Start at 0
        ([1, 2, 3, 4, 5], 2, [1, 2, 5, 4, 3]),  # Start at 2
        ([1, 2, 3, 4, 5], 4, [1, 2, 3, 4, 5]),  # Start at end of list
    ],
)
def test_reverse_from(input_list, start, expected_output):
    assert reverse_from(input_list, start) == expected_output


@pytest.mark.parametrize("input_list, start", [([1, 2, 3, 4, 5], -1), ([1, 2, 3, 4, 5], 5), ([1, 2, 3, 4, 5], 100)])
def test_reverse_from__out_of_bounds(input_list, start):
    with pytest.raises(IndexError):
        reverse_from(input_list, start)


@pytest.mark.parametrize(
    "input_list, expected_output",
    [
        ([], []),  # Empty list
        ([1], [1]),  # Single element
        ([1, 2, 3], [3, 2, 1]),  # Multiple elements
        ([1, 2, 3, 4], [4, 3, 2, 1]),  # Multiple elements
        ([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]),  # Multiple elements
    ],
)
def test_reverse_inplace(input_list, expected_output):
    reverse_inplace(input_list)
    assert input_list == expected_output
