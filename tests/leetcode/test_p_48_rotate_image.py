import pytest

from leetcode.p_48_rotate_image import rotate_image


@pytest.mark.parametrize(
    "input_data, expected",
    [
        (
            [[5, 1, 9, 11], [2, 4, 8, 10], [13, 3, 6, 7], [15, 14, 12, 16]],
            [[15, 13, 2, 5], [14, 3, 4, 1], [12, 6, 8, 9], [16, 7, 10, 11]],
        ),
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[7, 4, 1], [8, 5, 2], [9, 6, 3]]),
    ],
)
def test_function_name(input_data, expected):
    # the function under test rotates the matrix in_place
    rotate_image(input_data)
    assert input_data == expected
