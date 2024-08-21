import pytest

from leetcode.p_189_rotate_array import rotate_inplace_1, rotate_inplace_2


@pytest.mark.parametrize("func", [rotate_inplace_1, rotate_inplace_2])
@pytest.mark.parametrize(
    "input_data, k, expected",
    [([1, 2, 3, 4, 5, 6, 7], 3, [5, 6, 7, 1, 2, 3, 4]), ([-1, -100, 3, 99], 2, [3, 99, -1, -100])],
)
def test_function_name(func, input_data, k, expected):
    arr = input_data[:]
    func(arr, k)
    assert arr == expected
