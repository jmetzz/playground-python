def permut(input_sequence, level):
    return internal_permut(input_sequence, "", level)


def internal_permut(input_sequence, prefix, level):
    s = ""
    for _ in range(level):
        s += "\t"

    if len(input_sequence) == 0:
        print(f"{s}{prefix}")
    else:
        for idx, _ in enumerate(input_sequence):
            rem = input_sequence[:idx] + input_sequence[idx + 1 :]
            internal_permut(rem, prefix + input_sequence[idx], level + 1)


def is_permutation_by_sorting(input_sequence, other):
    """Checks if other is a valid permutation of the input

    Assumptions:
       - the method case sensitive
       - whitespaces are significant

    :param input_sequence the base string
    :param other the possible permutation of input

    :return bool
    """
    if len(input_sequence) != len(other):
        return False
    return sorted(input_sequence) == sorted(other)


def is_permutation_by_counting(input_sequence, other):
    """Checks if other is a valid permutation of the input

    Assumptions:
       - only ascii characters allowed
       - the method case sensitive
       - whitespaces are significant

    :param input_sequence the base string
    :param other the possible permutation of input

    :return bool
    """
    if len(input_sequence) != len(other):
        return False

    # b) count char appearance in both, using a index vector
    letters = [0] * 256
    for c in input_sequence:
        letters[ord(c)] += 1

    for _, element in enumerate(other):
        c = ord(element)
        letters[c] -= 1
        if letters[c] < 0:
            return False
    return True


def to_index(character) -> int:
    a, z = ord("a"), ord("z")
    value = ord(character)
    if a <= value <= z:
        return value - a
    return -1


def is_permut_of_palindrome(input_sequence: str) -> bool:
    """Checks if a string is a permutation a palindrome

    :param input_sequence string
    :return bool

    Definitions:
        - strings of even length should have even number of individual
        characters.
        - string of an odd length, must have exactly one character with
        an odd count
    """

    def histogram_table(input_sequence: str) -> list:
        """Maps each character in input to a number

        Non letter character maps to -1
        """
        table = [0] * (ord("z") - ord("a") + 1)
        for c in input_sequence:
            idx = to_index(c)
            if idx != -1:
                table[idx] += 1
        return table

    def has_max_one_odd(table: list) -> bool:
        found_odd = False
        for count in table:
            if count % 2 == 1:
                if found_odd:
                    return False
                found_odd = True
        return True

    alphabet_table = histogram_table(input_sequence)
    return has_max_one_odd(alphabet_table)


def is_permut_of_palindrome_2(input_sequence: str) -> bool:
    """Checks if a string is a permutation a palindrome

    :param input_sequence string
    :return bool

    Definitions:
        - strings of even length should have even number of individual
        characters.
        - string of an odd length, must have exactly one character with
        an odd count

    Change: checks the number of odd number as we go
    """
    table = [0] * (ord("z") - ord("a") + 1)
    odd_count = 0
    for c in input_sequence:
        idx = to_index(c)
        if idx != -1:
            table[idx] += 1
            if table[idx] % 2 == 1:
                odd_count += 1
            else:
                odd_count -= 1

    return odd_count <= 1


def is_permut_of_palindrome_3(input_sequence: str) -> bool:
    """Checks if a string is a permutation a palindrome

    :param input_sequence string
    :return bool

    Definitions:
        - strings of even length should have even number of individual
        characters.
        - string of an odd length, must have exactly one character with
        an odd count

    Change: uses bit vector and ignore the frequency. We only need to know
    if the number of odd letter is more than one
    """

    def create_bit_vector(input_sequence: str) -> int:
        bit_vector = 0
        for c in input_sequence:
            bit_vector = toggle(bit_vector, to_index(c))
        return bit_vector

    def toggle(bit_vector: int, index: int) -> int:
        """Toggle the i-th bit in the integer"""
        if index < 0:
            return bit_vector

        # mask keeps the i-th bit set as 1 and all other as 0
        mask = 1 << index

        if (bit_vector & mask) == 0:
            bit_vector |= mask  # set the bit
        else:
            bit_vector &= ~mask  # clear the bit
        return bit_vector

    def check_exactly_one_bit_set(bit_vector) -> bool:
        """Check that at most ont bit in the integer is set to 1

        A very elegant way to check that an integer has exactly one bit set to 1.

        Picture an integer like 00010000. Subtract 1 from the number,
        we'll get 00001111. Note that there is no overlap between the bits
        in these numbers. So, we can check to see that a number has exactly
        one 1 because if we subtract 1 from it and then apply the AND operator
        between the two numbers, we should get 0.

        :param bit_vector:
        :return: bool
        """
        return (bit_vector & (bit_vector - 1)) == 0

    bits = create_bit_vector(input_sequence)
    return bits == 0 or check_exactly_one_bit_set(bits)


if __name__ == "__main__":
    # print(f"All permutations of 'learn': {permut('learn', 0)}")

    # print("Using sorting solution")
    # print(f"Is 'abc' permutation of 'cba'? "
    #       f"{is_permutation_by_sorting('abc', 'cba')}")
    # print(f"Is 'abc' permutation of 'bca'? "
    #       f"{is_permutation_by_sorting('abc', 'bca')}")
    # print(f"Is 'abcc' permutation of 'cba'? "
    #       f"{is_permutation_by_sorting('abcc', 'cba')}")

    # print("Using counting solution")
    # print(f"Is 'abc' permutation of 'cba'? "
    #       f"{is_permutation_by_counting('abc', 'cba')}")
    # print(f"Is 'abc' permutation of 'bca'? "
    #       f"{is_permutation_by_counting('abc', 'bca')}")
    # print(f"Is 'abcc' permutation of 'cba'? "
    #       f"{is_permutation_by_counting('abcc', 'cba')}")

    print("\nsolution 1")
    print(is_permut_of_palindrome("adciicda"))
    print(is_permut_of_palindrome("adcidca"))
    print(is_permut_of_palindrome("adciicdaa"))
    print(is_permut_of_palindrome("addciicdaa"))

    print("\nsolution 2")
    print(is_permut_of_palindrome_2("adciicda"))
    print(is_permut_of_palindrome_2("adcidca"))
    print(is_permut_of_palindrome_2("adciicdaa"))
    print(is_permut_of_palindrome_2("addciicdaa"))

    print("\nsolution 3")
    print(is_permut_of_palindrome_3("adciicda"))
    print(is_permut_of_palindrome_3("adcidca"))
    print(is_permut_of_palindrome_3("adciicdaa"))
    print(is_permut_of_palindrome_3("addciicdaa"))
