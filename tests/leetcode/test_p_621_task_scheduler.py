import pytest
from leetcode.p_621_task_scheduler import (
    least_interval_chunck_based,
    least_interval_max_heap,
    least_interval_max_heap_2,
)

test_cases = [
    ([], 2, 0),  # No tasks
    (["A"], 2, 1),  # Single task
    (["A", "B", "C", "A", "B"], 1, 5),  # Tasks can be scheduled without idle times
    (["A", "A", "A", "B", "B", "B"], 2, 8),  # Basic case with cooling period
    (["A", "C", "A", "B", "D", "B"], 1, 6),  # Tasks can be scheduled without idle times
    (["A", "A", "A", "B", "B", "B"], 3, 10),
    (["A", "A", "A", "B", "B", "B", "C", "C", "D", "D"], 2, 10),
]


@pytest.mark.parametrize("tasks, n, expected", test_cases)
def test_least_interval_max_heap(tasks, n, expected):
    assert least_interval_max_heap(tasks, n) == expected


@pytest.mark.parametrize("tasks, n, expected", test_cases)
def test_least_interval_max_heap_2(tasks, n, expected):
    assert least_interval_max_heap_2(tasks, n) == expected


@pytest.mark.parametrize("tasks, n, expected", test_cases)
def test_least_interval_chunck_based(tasks, n, expected):
    assert least_interval_chunck_based(tasks, n) == expected
