import pytest

from leetcode.p_73_set_matrix_zeroes import set_zeroes_1, set_zeroes_2


@pytest.mark.parametrize(
    "input_data, expected",
    [
        ([[1, 1, 1], [1, 0, 1], [1, 1, 1]], [[1, 0, 1], [0, 0, 0], [1, 0, 1]]),
        ([[0, 1, 2, 0], [3, 4, 5, 2], [1, 3, 1, 5]], [[0, 0, 0, 0], [0, 4, 5, 0], [0, 3, 1, 0]]),
    ],
)
def test_set_zeros_1(input_data, expected):
    # the functions under test change input_data in place
    set_zeroes_1(input_data)
    assert input_data == expected


@pytest.mark.parametrize(
    "input_data, expected",
    [
        ([[1, 1, 1], [1, 0, 1], [1, 1, 1]], [[1, 0, 1], [0, 0, 0], [1, 0, 1]]),
        ([[0, 1, 2, 0], [3, 4, 5, 2], [1, 3, 1, 5]], [[0, 0, 0, 0], [0, 4, 5, 0], [0, 3, 1, 0]]),
    ],
)
def test_set_zeros_2(input_data, expected):
    # the functions under test change input_data in place
    set_zeroes_2(input_data)
    assert input_data == expected
