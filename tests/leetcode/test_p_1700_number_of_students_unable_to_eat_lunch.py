import pytest
from leetcode.p_1700_number_of_students_unable_to_eat_lunch import count_students_counter, count_students_queue


@pytest.mark.parametrize("func", [count_students_queue, count_students_counter])
@pytest.mark.parametrize(
    "students, sandwiches, expected", [([1, 1, 1, 0, 0, 1], [1, 0, 0, 0, 1, 1], 3), ([1, 1, 0, 0], [0, 1, 0, 1], 0)]
)
def test_count_students(func, students, sandwiches, expected):
    assert func(students, sandwiches) == expected
