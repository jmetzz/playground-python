from leetcode.p_155_min_stack import MinStack


def test_min_stack_operations():
    stack = MinStack()

    # Test initial state
    assert stack.top() is None, "Top of empty stack should be None"
    assert stack.min() is None, "Min of empty stack should be None"

    # Test push and min
    stack.push(5)
    assert stack.top() == 5, "Top should be 5 after pushing 5"
    assert stack.min() == 5, "Min should be 5 after pushing 5"

    stack.push(3)
    assert stack.top() == 3, "Top should be 3 after pushing 3"
    assert stack.min() == 3, "Min should be 3 after pushing 3"

    stack.push(7)
    assert stack.top() == 7, "Top should be 7 after pushing 7"
    assert stack.min() == 3, "Min should still be 3 after pushing 7"

    stack.push(2)
    assert stack.top() == 2, "Top should be 2 after pushing 2"
    assert stack.min() == 2, "Min should be 2 after pushing 2"

    # Test pop and min
    stack.pop()
    assert stack.top() == 7, "Top should be 7 after popping 2"
    assert stack.min() == 3, "Min should revert to 3 after popping 2"

    stack.pop()
    assert stack.top() == 3, "Top should be 3 after popping 7"
    assert stack.min() == 3, "Min should still be 3 after popping 7"

    stack.pop()
    assert stack.top() == 5, "Top should be 5 after popping 3"
    assert stack.min() == 5, "Min should revert to 5 after popping 3"

    stack.pop()
    assert stack.top() is None, "Top should be None after popping the last element"
    assert stack.min() is None, "Min should be None after popping the last element"

    assert stack.is_empty(), "Should be empty by now"
