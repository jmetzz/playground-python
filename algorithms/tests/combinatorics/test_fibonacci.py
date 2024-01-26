import pytest

from dynamic_programming.p_1_fibonacci import fib_recursive, fib_memoization, fib_sequence, \
    fib_sequence_via_generator, fib_memoization


@pytest.mark.parametrize("test_input, expected",
                         [
                             (0, 0),
                             (1, 1),
                             (2, 1),
                             (3, 2),
                             (4, 3),
                             (7, 13),
                             (8, 21)
                         ])
def test_fib_recursive(test_input, expected):
    assert fib_recursive(test_input) == expected


@pytest.mark.parametrize("test_input, expected",
                         [
                             (0, 0),
                             (1, 1),
                             (2, 1),
                             (3, 2),
                             (4, 3),
                             (7, 13),
                             (8, 21)
                         ])
def test_fib_with_memoization(test_input, expected):
    # assert fib_with_memoization(test_input) == expected
    assert fib_memoization(test_input, {}) == expected


@pytest.mark.parametrize("test_input, expected",
                         [
                             (0, [0]),
                             (1, [0, 1]),
                             (2, [0, 1, 1]),
                             (3, [0, 1, 1, 2]),
                             (4, [0, 1, 1, 2, 3]),
                             (7, [0, 1, 1, 2, 3, 5, 8, 13]),
                             (8, [0, 1, 1, 2, 3, 5, 8, 13, 21])
                         ])
def test_fib_sequence(test_input, expected):
    assert fib_sequence(test_input) == expected


@pytest.mark.parametrize("test_input, expected",
                         [
                             (0, [0]),
                             (1, [0, 1]),
                             (2, [0, 1, 1]),
                             (3, [0, 1, 1, 2]),
                             (4, [0, 1, 1, 2, 3]),
                             (7, [0, 1, 1, 2, 3, 5, 8, 13]),
                             (8, [0, 1, 1, 2, 3, 5, 8, 13, 21])
                         ])
def test_fib_sequence_via_generator(test_input, expected):
    assert list(fib_sequence_via_generator(test_input)) == expected
