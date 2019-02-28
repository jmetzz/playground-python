def all_unique_chars(input) -> bool:
    """Checks if all the characters are unique

    Assuming extended ascii string and alphabet of up to 256 characters
    :param input the ascii string
    :return bool
    """
    if len(input) > 256:
        return False

    alphabet = [False] * 256

    for c in input:
        idx = ord(c)
        if alphabet[idx]:
            return False
        else:
            alphabet[idx] = True
    return True


def all_unique_chars_bit_op(input) -> bool:
    """Checks if all the characters are unique

    Assuming extended ascii string and alphabet of up to 256 characters
    :param input the ascii string
    :return bool
    """
    checker = 0
    for i in range(len(input)):
        val = ord(input[i])
        if (checker & (1 << val)) > 0:
            return False
        else:
            checker |= (1 << val)
    return True


if __name__ == "__main__":
    print("alphabet vector solution:")
    print(f"unique chars in 'ABCD'? : {all_unique_chars('ABCD')}")
    print(f"unique chars in 'ABCDA'? : {all_unique_chars('ABCDA')}")

    print("bit vector solution:")
    print(f"unique chars in 'ABCD'? : {all_unique_chars_bit_op('ABCD')}")
    print(f"unique chars in 'ABCDA'? : {all_unique_chars_bit_op('ABCDA')}")
    print(f"unique chars in 'abcd'? : {all_unique_chars_bit_op('abcd')}")
    print(f"unique chars in 'abcdA'? : {all_unique_chars_bit_op('abcdA')}")
    print(f"unique chars in 'abcdAb'? : {all_unique_chars_bit_op('abcdAb')}")
