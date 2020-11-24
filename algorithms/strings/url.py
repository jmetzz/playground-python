def as_url(sequence, true_length) -> str:
    """Replace all spaces in a string with '%20' symbols

        Assumptions:
        - using an array of characters so that we can make changes in place
        - the array has enough space to hold the modified input
        - you are given the true size of the string

        Args:
            sequence: of characters
            true_length: reflects the true size of the sequence array,
            even the white spaces

        Returns:
             a string with spaces replaced

        Example:
            > as_url("John  Doe     ", 8)
            > John%20Doe%20

    """
    # count the number of whitespaces
    # in the 'true_length' boundaries
    white_space_counter = 0
    for i in range(true_length):
        if sequence[i] == " ":
            white_space_counter += 1

    # apply the modification from right to left
    index = true_length + white_space_counter * 2
    for i in range(true_length - 1, -1, -1):
        if sequence[i] == " ":
            sequence[index - 1] = "0"
            sequence[index - 2] = "2"
            sequence[index - 3] = "%"
            index -= 3
        else:
            sequence[index - 1] = sequence[i]
            index -= 1
    return sequence


if __name__ == "__main__":
    print("".join(as_url(list("john doe   "), 8)))
