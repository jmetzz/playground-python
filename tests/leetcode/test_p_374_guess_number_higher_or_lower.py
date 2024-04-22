import pytest
from leetcode.p_374_guess_number_higher_or_lower import GuessNumber

test_cases = [(6, 10), (1, 1), (1, 2)]


@pytest.mark.parametrize("input_pick, input_n", test_cases)
def test_guess_number_recursive(input_pick, input_n):
    assert GuessNumber(input_pick).guess_number_recursive(input_n) == input_pick


@pytest.mark.parametrize("input_pick, input_n", test_cases)
def test_guess_number_iterative(input_pick, input_n):
    assert GuessNumber(input_pick).guess_number_iterative(input_n) == input_pick
