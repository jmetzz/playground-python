import pytest

from leetcode.p_232_queue_using_stacks import QueueUsingStacks


def test_is_empty__init():
    queue = QueueUsingStacks()
    assert queue.empty()


def test_is_not_empty():
    queue = QueueUsingStacks()
    queue.push(1)
    assert not queue.empty()


def test_push_one():
    queue = QueueUsingStacks()
    queue.push(1)
    assert queue._stack == [1]


def test_push_two():
    queue = QueueUsingStacks()
    queue.push(1)
    queue.push(2)
    assert queue._stack == [1, 2]


def test_peek():
    queue = QueueUsingStacks()
    queue.push(1)
    queue.push(2)
    assert queue.peek() == 1


def test_queue_using_stacks():
    # Your MyQueue object will be instantiated and called as such:
    queue = QueueUsingStacks()
    queue.push(1)
    queue.push(2)
    assert queue.peek() == 1
    assert queue.pop() == 1
    assert not queue.empty()
