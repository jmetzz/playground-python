"""Considering a 2d grid of m x n, how many ways can you travel from the top left corner for the bottom right corner?
The only actions allowed are:
1. move down
2. move right

"""


def solve_tabulation(m: int, n: int) -> int:
    pass


if __name__ == "__main__":
    memo = dict()
    value = solve_tabulation(2, 3)
    print(value)
    print(memo)

    value = solve_tabulation(18, 18)  # 2333606220
    print(value)
