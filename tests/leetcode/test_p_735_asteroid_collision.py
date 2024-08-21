import pytest

from leetcode.p_735_asteroid_collision import asteroid_collision, asteroid_collision_2


@pytest.mark.parametrize("func", [asteroid_collision, asteroid_collision_2])
@pytest.mark.parametrize(
    "asteroids, expected", [([8, -8], []), ([5, 10, -5], [5, 10]), ([10, 2, -5], [10]), ([-1, 3, 2, -3], [-1])]
)
def test_asteroid_collision(func, asteroids, expected):
    assert func(asteroids) == expected
