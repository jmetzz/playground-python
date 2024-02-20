import pytest
from leetcode.p_1642_furthest_building_you_can_reach import FurthestBuilding


@pytest.mark.parametrize(
    "input_heights, input_bricks, input_ladders, expected",
    [
        ([4, 2, 7, 6, 9, 14, 12], 5, 1, 4),
        ([4, 12, 2, 7, 3, 18, 20, 3, 19], 10, 2, 7),
        ([14, 3, 19, 3], 17, 0, 3),
        ([1, 5, 1, 2, 3, 4, 10000], 4, 1, 5),
    ],
)
def test_furthest_building(input_heights, input_bricks, input_ladders, expected):
    assert FurthestBuilding().solve(input_heights, input_bricks, input_ladders) == expected
