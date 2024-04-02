import pytest
from leetcode.p_933_number_of_recent_calls import RecentCounter


@pytest.fixture
def test_cases():
    return [
        (1, 1),
        (100, 2),
        (3001, 3),
        (3002, 3),
        (3003, 4),
        (4000, 4),
        (5000, 5),
        (6000, 6),
        (8000, 3),
    ]


def test_recent_counter(test_cases):
    counter = RecentCounter()
    for ping_even, expected in test_cases:
        assert counter.ping(ping_even) == expected, f"Failed at ping({ping_even})"


def test_recent_counter_queue_state(test_cases):
    counter = RecentCounter()
    for event, _ in test_cases:
        counter.ping(event)
    assert list(counter._q) == [5000, 6000, 8000], "The queue state is not as expected after all pings."
