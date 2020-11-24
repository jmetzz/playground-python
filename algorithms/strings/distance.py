def one_away(firsts: str, second: str) -> bool:
    """Given two strings, write a function to check if they are one edit (or zero edits) operation away.
        There are three types of edits that can be performed on strings:
        - insert a character,
        - remove a character, or
        - replace a character.
    :param a string
    :param b string
    :return bool
    """
    if len(firsts) == len(second):
        return one_replace_away(firsts, second)
    elif len(firsts) + 1 == len(second):
        return one_insert_away(firsts, second)
    else:
        return one_insert_away(second, firsts)
    return False


def one_replace_away(first: str, second: str) -> bool:
    found_diff = False
    for idx in range(len(first)):
        if first[idx] != second[idx]:
            if found_diff:
                return False
            found_diff = True
    return found_diff


def one_insert_away(first: str, second: str) -> bool:
    idx_first = 0
    idx_second = 0
    while idx_first < len(first) and idx_second < len(second):
        if first[idx_first] == second[idx_second]:
            idx_first += 1
            idx_second += 1
        else:
            if idx_first != idx_second:
                # have already found the diff point before
                return False
            # otherwise, advance the pointer on the second string
            idx_second += 1
    return True


if __name__ == "__main__":
    print(one_away("pale", "ple"))
