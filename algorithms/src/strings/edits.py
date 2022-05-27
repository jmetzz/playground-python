def insert(sequence: str, idx: int, element) -> str:
    if idx < 0:
        raise ValueError("Only positive indices are valid")

    if idx == 0:
        return element + sequence
    return sequence[: idx - 1] + element + sequence[idx:]


def remove(sequence: str, idx: int) -> str:
    return sequence[:idx] + sequence[idx + 1 :]


def replace(sequence: str, idx: int, element) -> str:
    # there is also a builtin function to accomplish
    # string modification: s.replace('b', 'x')
    # but this function is not index based.
    return sequence[:idx] + element + sequence[idx + 1 :]


if __name__ == "__main__":
    print("Original: pale")
    input_sequence = "pale"
    input_sequence = insert(input_sequence, 0, "p")
    print(f"After insertion of p at 2: {input_sequence}")

    input_sequence = remove(input_sequence, 0)
    print(f"After removing element at 0: {input_sequence}")

    input_sequence = replace(input_sequence, 2, "t")
    print(f"After replacing element 2 by 't': {input_sequence}")
