def insert(s: str, idx: int, c) -> str:
    if idx < 0:
        raise ValueError("Only positive indices are valid")

    if idx == 0:
        return c + s
    else:
        return s[:idx - 1] + c + s[idx:]


def remove(s: str, idx: int) -> str:
    return s[:idx] + s[idx + 1:]


def replace(s: str, idx: int, c) -> str:
    # there is also a builtin function to accomplish
    # string modification: s.replace('b', 'x')
    # but this function is not index based.
    return s[:idx] + c + s[idx + 1:]

if __name__ == "__main__":
    print('Original: pale')
    s = 'pale'
    s = insert(s, 0, 'p')
    print(f"After insertion of p at 2: {s}")

    s = remove(s, 0)
    print(f"After removing element at 0: {s}")

    s = replace(s, 2, 't')
    print(f"After replacing element 2 by 't': {s}")
