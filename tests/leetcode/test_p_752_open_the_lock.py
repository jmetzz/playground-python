import pytest
from leetcode.p_752_open_the_lock import open_lock_str, open_lock_tuple


@pytest.mark.parametrize("func", [open_lock_str, open_lock_tuple])
@pytest.mark.parametrize(
    "input_deadlocks, target, expected",
    [
        (["0201", "0101", "0102", "1212", "2002"], "0202", 6),
        (["8888"], "0009", 1),
        (["8887", "8889", "8878", "8898", "8788", "8988", "7888", "9888"], "8888", -1),
    ],
)
def test_open_lock(func, input_deadlocks, target, expected):
    assert func(input_deadlocks, target) == expected
