import pytest
from leetcode.p_79_word_search import exist_1, exist_2, exist_3

grid_1 = [["A", "B", "C", "E"], ["S", "F", "C", "S"], ["A", "D", "E", "E"]]
grid_2 = [["A", "B", "C", "E"], ["S", "F", "E", "S"], ["A", "D", "E", "E"]]


test_cases = [
    (grid_1, "ABCCED", True),
    (grid_1, "SEE", True),
    (grid_1, "ABCB", False),
    (grid_2, "ABCESEEEFS", True),
    (grid_2, "ABCEFSADEESE", True),
]


@pytest.mark.parametrize("func", [exist_1, exist_2, exist_3])
@pytest.mark.parametrize("grid, word, expected", test_cases)
def test_word_search(func, grid, word, expected):
    assert func(grid, word) == expected
