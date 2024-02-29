from typing import Dict


def is_palindrome_with_stack(sequence: str, cache: Dict[str, bool]) -> bool:
    if sequence in cache:
        return cache[sequence]

    n = len(sequence)
    if n == 1:
        return True

    stack = []
    i = 0
    mid_idx = n // 2
    while i < mid_idx:
        stack.append(sequence[i])
        i += 1

    i = mid_idx if n % 2 == 0 else mid_idx + 1
    while stack and stack.pop() == sequence[i]:
        i += 1

    cache[sequence] = i == n
    return cache[sequence]


def is_palindrome_expanding_window(sequence: str) -> bool:
    n = len(sequence)
    if n == 1:
        return True

    right = n // 2
    left = right - 1 if n % 2 == 0 else right
    while left >= 0 and right < len(sequence) and sequence[left] == sequence[right]:
        left, right = left - 1, right + 1

    return right >= len(sequence)


def is_palindrome_1(sequence: str) -> bool:
    n = len(sequence)
    mid = n // 2
    return sequence[:mid] == (sequence[mid:] if n % 2 == 0 else sequence[mid + 1 :])[::-1]


def is_palindrome_2(sequence: str) -> bool:
    return sequence == sequence[::-1]
