import pytest

from leetcode.p_169_majority_element import MajorityElement

INPUT_1 = []
INPUT_2 = [1]
INPUT_3 = [3, 2, 3]
INPUT_4 = [2, 2, 1, 1, 1, 2, 2]
INPUT_5 = [6, 6, 6, 7, 7]
INPUT_6 = [1, 2, 3]
INPUT_7 = [6, 6, 6, 7, 7, 7]


@pytest.mark.parametrize("test_input, expected",
                         [
                             (INPUT_1, None),
                             (INPUT_2, 1),
                             (INPUT_3, 3),
                             (INPUT_4, 2),
                             (INPUT_5, 6),
                             (INPUT_6, None),
                             (INPUT_6, None)
                         ])
def test_majority_element_solve_counter(test_input, expected):
    assert MajorityElement().solve_counter(test_input) == expected


@pytest.mark.parametrize("test_input, expected",
                         [
                             (INPUT_1, None),
                             (INPUT_2, 1),
                             (INPUT_3, 3),
                             (INPUT_4, 2),
                             (INPUT_5, 6),
                             (INPUT_6, None),
                             (INPUT_6, None)
                         ])
def test_majority_element_solve_heap(test_input, expected):
    assert MajorityElement().solve_heap(test_input) == expected


@pytest.mark.parametrize("test_input, expected",
                         [
                             (INPUT_1, None),
                             (INPUT_2, 1),
                             (INPUT_3, 3),
                             (INPUT_4, 2),
                             (INPUT_5, 6),
                             (INPUT_6, None),
                             (INPUT_6, None)
                         ])
def test_majority_element_solve_boyer_moore(test_input, expected):
    assert MajorityElement().solve_boyer_moore_majority_alg(test_input) == expected
