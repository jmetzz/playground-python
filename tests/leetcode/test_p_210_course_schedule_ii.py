import pytest

from leetcode.p_210_course_schedule_ii import find_order


@pytest.mark.parametrize(
    ("num_courses", "prerequisites", "expected"),
    [
        (1, [], [0]),
        (2, [[1, 0]], [0, 1]),
        (4, [[1, 0], [2, 0], [3, 1], [3, 2]], [0, 1, 2, 3]),
    ],
)
def test_function_name(num_courses, prerequisites, expected):
    assert find_order(num_courses, prerequisites) == expected
