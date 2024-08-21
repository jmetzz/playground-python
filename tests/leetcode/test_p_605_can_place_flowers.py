import pytest

from leetcode.p_605_can_place_flowers import can_place_flowers_naive, can_place_flowers_padding


@pytest.mark.parametrize("func", [can_place_flowers_naive, can_place_flowers_padding])
@pytest.mark.parametrize(
    "input_arr, n, expected",
    [
        ([1, 0, 0, 0, 1], 1, True),
        ([1, 0, 0, 0, 0, 0, 1], 2, True),
        ([0, 0, 1, 0, 1], 1, True),
        ([1, 0, 1, 0, 1, 0, 1], 0, True),
        ([1, 0, 0, 0, 1], 2, False),
        ([1, 0, 0, 0, 0, 1], 2, False),
    ],
)
def test_can_place_flowers(func, input_arr, n, expected):
    assert func(input_arr[:], n) == expected
