CHARS_MAP = {
    "2": "abc",
    "3": "def",
    "4": "ghi",
    "5": "jkl",
    "6": "mno",
    "7": "pqrs",
    "8": "tuv",
    "9": "wxyz",
}


def letter_combinations_iterative(digits: str) -> list[str]:
    # initialization, if digits is empty, return []
    combinations = [""] if digits else []
    for d in digits:
        combinations = [p + q for p in combinations for q in CHARS_MAP[d]]
    return combinations


def letter_combinations_backtracking(digits: str) -> list[str]:
    size = len(digits)
    if size == 0:
        return []
    answer = []

    def backtrack(idx: int, curr_str):
        # base
        if idx == size:
            # have already processed all elements
            answer.append(curr_str)
            return
        # iterate over all the options
        options = CHARS_MAP[digits[idx]]
        for ch in options:
            backtrack(idx + 1, curr_str + ch)

    # trigger the function starting from position 0,
    # to accumulate the combinations into answer array
    backtrack(0, "")
    return answer


if __name__ == "__main__":
    inputs = [
        ("23", ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"]),
        ("", []),
        ("2", ["a", "b", "c"]),
    ]
    for digits, _ in inputs:
        print(letter_combinations_iterative(digits))
        print(letter_combinations_backtracking(digits))
        print()
