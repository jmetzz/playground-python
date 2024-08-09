def reverse(elements: list) -> list:
    return elements[::-1]


def reverse_from(elements: list, start: int = 0) -> list:
    if len(elements) == 0:
        return []

    if start < 0 or start >= len(elements):
        raise IndexError(f"Start index out of bounds: '{start}'")

    if start == 0:
        return elements[::-1]

    return elements[0:start] + elements[: start - 1 : -1]


def reverse_inplace(elements: list, start: int = 0) -> None:
    left, right = start, len(elements) - 1
    while left < right:
        elements[left], elements[right] = elements[right], elements[left]
        left += 1
        right -= 1


if __name__ == "__main__":
    arr = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    print(f"Original: {arr}")
    print(f"reverse: {reverse(arr)}")
    print(f"reverse from idx 0: {reverse_from(arr, start=0)}")
    print(f"reverser from idx 5: {reverse_from(arr, start=5)}")
    reverse_inplace(arr)
    print(f"after reverser inplace: {arr}")
