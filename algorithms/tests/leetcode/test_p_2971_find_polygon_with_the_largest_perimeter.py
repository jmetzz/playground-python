import pytest

from leetcode.p_2971_find_polygon_with_the_largest_perimeter import FindPolygonWithLargestPerimeter


@pytest.mark.parametrize("test_input, expected",
                         [
                             ([5, 5, 5], 15),
                             ([1, 12, 1, 2, 5, 50, 3], 12),
                             ([5, 5, 50], -1),
                             ([143, 24, 3, 9, 2, 18, 3, 22, 15], 96),
                         ])
def test_find_polygon_largest_perimeter_loop(test_input, expected):
    assert FindPolygonWithLargestPerimeter().solve_with_loop(test_input) == expected


@pytest.mark.parametrize("test_input, expected",
                         [
                             ([5, 5, 5], 15),
                             ([1, 12, 1, 2, 5, 50, 3], 12),
                             ([5, 5, 50], -1),
                             ([143, 24, 3, 9, 2, 18, 3, 22, 15], 96),
                         ])
def test_find_polygon_largest_perimeter_heap(test_input, expected):
    assert FindPolygonWithLargestPerimeter().solve_with_heap(test_input) == expected


@pytest.mark.parametrize("test_input, expected",
                         [
                             ([5, 5, 5], 15),
                             ([1, 12, 1, 2, 5, 50, 3], 12),
                             ([5, 5, 50], -1),
                             ([143, 24, 3, 9, 2, 18, 3, 22, 15], 96),
                         ])
def test_find_polygon_largest_perimeter_reversed(test_input, expected):
    assert FindPolygonWithLargestPerimeter().solve_reversed(test_input) == expected
