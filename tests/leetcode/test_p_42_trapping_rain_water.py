import pytest

from leetcode.p_42_trapping_rain_water import trapped_water_volume_monotonic_stack, trapped_water_volume_two_pointers


@pytest.mark.parametrize("func", [trapped_water_volume_two_pointers, trapped_water_volume_monotonic_stack])
@pytest.mark.parametrize(
    "input_heights, expected",
    [
        ([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1], 6),
        ([4, 2, 0, 3, 2, 5], 9),
    ],
)
def test_trapped_water_volume(func, input_heights, expected):
    assert func(input_heights) == expected
