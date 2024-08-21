import pytest

from leetcode.p_118_pascal_triangle import generate_iterative, generate_iterative_padding, generate_recursive


@pytest.mark.parametrize("func", [generate_iterative, generate_iterative_padding, generate_recursive])
@pytest.mark.parametrize(
    "num_rows, expected",
    [
        (0, []),
        (1, [[1]]),
        (2, [[1], [1, 1]]),
        (5, [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1]]),
    ],
)
def test_pascal_triangle_generation(func, num_rows, expected):
    assert func(num_rows) == expected
