from typing import List


def compress(chars: List[str]) -> int:
    n = len(chars)
    right = 0
    left = 0

    while right < n:
        count = 0
        curr_char = chars[right]
        # count repeating curr_char
        while right < n and chars[right] == curr_char:
            right += 1
            count += 1
        # add the count of curr_char and respective count to chars array
        chars[left] = curr_char
        left += 1
        if count > 1:
            str_count = str(count)
            num_digits = len(str_count)
            chars[left : left + num_digits] = list(str_count)
        else:
            num_digits = 0
        # advance left pointer to the correct position for the next char
        left += num_digits
    return left


if __name__ == "__main__":
    arr = ["a", "a", "b", "b", "c", "c", "c"]
    print(compress(arr))
    print(arr)

    arr = ["a"]
    print(compress(arr))
    print(arr)

    arr = ["a", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b"]
    print(compress(arr))
    print(arr)

    arr = ["a", "a", "a", "b", "b", "a", "a"]
    print(compress(arr))
    print(arr)
