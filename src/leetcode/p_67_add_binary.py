"""https://leetcode.com/problems/add-binary/description/

67. Add Binary
Easy

Given two binary strings a and b, return their sum as a binary string.


Example 1:
Input: a = "11", b = "1"
Output: "100"

Example 2:
Input: a = "1010", b = "1011"
Output: "10101"


Constraints:

1 <= a.length, b.length <= 104
a and b consist only of '0' or '1' characters.
Each string does not contain leading zeros except for the zero itself.
"""


def add_binary(a: str, b: str) -> str:
    """Adds two binary numbers represented as strings and returns the sum as a binary string.

    The function iterates from the least significant bit to the most significant bit of each input string,
    adding corresponding bits along with any carry from the previous addition.

    The result is constructed in reverse order and then reversed at the end to match the
    conventional binary representation.

    Approach:
        - Use two pointers to iterate through each string from right to left.
        - At each step, add the corresponding bits and the carry from the previous step.
        - Calculate the new carry by dividing the sum by 2 (using integer division).
        - Append the remainder (sum modulo 2) to the answer.
        - After processing both strings, check if there's an outstanding carry and add it to the answer.
        - Reverse the answer string to represent the binary number correctly.

    Rationale:
    In binary arithmetic, when you add two bits, the resulting sum can only be in the range of 0 to 2.
    This is because the maximum sum occurs when both bits are 1 (1 + 1 = 2).
    The use of % 2 (modulo 2) and // 2 (integer division by 2) in handling the carry during binary addition
    is based on how binary numbers are represented and how addition works in binary.

    * Using % 2 for the Resulting Bit - the % 2 operation finds the remainder when the sum of two bits
    (plus any carry from a previous operation) is divided by 2. This is equivalent to determining
    the least significant bit of the binary sum:
        If the sum is 0 or 2, % 2 results in 0.
        If the sum is 1, % 2 results in 1.
    This operation effectively results the bit value that should be added to the answer at the current position.

    * Using // 2 for the Carry - the // 2 operation performs integer division by 2, which is used to determine
    the carry for the next addition. In binary addition:
        A sum of 0 or 1 means there's no carry, as these values can be directly represented in binary with a single bit.
        A sum of 2 generates a carry because 2 in binary is represented as 10. The 1 is the carry that needs
        to be added to the next significant bit's sum.
    Thus, // 2 tells us whether the sum of two bits (plus any carry) generates a new carry for the next bit position.
    If the sum is 2, // 2 equals 1, indicating a carry. If the sum is 0 or 1, // 2 equals 0, indicating no carry.

    Args:
    ----
        a (str): The first binary number as a string.
        b (str): The second binary number as a string.

    Returns:
    -------
        str: The sum of the two binary numbers as a binary string.


    """
    answer = []
    carry = 0
    idx_a = len(a) - 1
    idx_b = len(b) - 1

    while idx_a >= 0 or idx_b >= 0 or carry:
        if idx_a >= 0:
            carry += int(a[idx_a])
            idx_a -= 1
        if idx_b >= 0:
            carry += int(b[idx_b])
            idx_b -= 1
        answer.append(str(carry % 2))
        carry //= 2
    return "".join(answer[::-1])


def add_binary_cheeting(a: str, b: str) -> str:
    return bin(int(a, 2) + int(b, 2))[2::]
