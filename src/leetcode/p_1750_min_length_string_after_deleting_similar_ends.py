def minimum_length_recursive(sequence: str) -> int:
    def helper(low, high):
        if low == high:
            return 1
        elif low > high:
            return 0
        if sequence[low] != sequence[high]:
            return high - low + 1

        while low < high and sequence[low] == sequence[low + 1]:
            low += 1
        while low < high and sequence[high - 1] == sequence[high]:
            high -= 1

        return helper(low + 1, high - 1)

    return helper(0, len(sequence) - 1)


def minimum_length_loop(sequence: str) -> int:
    low, high = 0, len(sequence) - 1
    while low < high and sequence[low] == sequence[high]:
        ch = sequence[low]
        while low <= high and sequence[low] == ch:
            low += 1
        while low <= high and sequence[high] == ch:
            high -= 1
    return high - low + 1


if __name__ == "__main__":
    print(minimum_length_loop("bbbbbbbbbbbbbbbbbbbbbbbbbbbabbbbbbbbbbbbbbbccbcbbcbccbbabbb"))
    # bbbbbbbbbbbbbbbbbbbbbbbbbbbabbbbbbbbbbbbbbbccbcbbcbccbbabbb -> 0
    # "bbbbbbbbbbbbbbbbbbbbbbbbbbbabbbbbbbbbbbbbbbccbcbcbccbbabbb" -> 1
    # "bbbbbbbbbbbbbbbbbbbbbbbbbbbabbbbbbbbbbbbbbbccbcbcbccbbabbb"
    # "                           abbbbbbbbbbbbbbbccbcbcbccbba   "
    # "                            bbbbbbbbbbbbbbbccbcbcbccbb    "
    # "                                           ccbcbcbcc      "
    # "                                             bcbcb        "
    # "                                              cbc         "
    # "                                               b          "
