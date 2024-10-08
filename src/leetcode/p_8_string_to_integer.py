"""https://leetcode.com/problems/sequence-to-integer-atoi/description/

8. sequence to Integer (atoi)
Medium

Implement the myAtoi(sequence s) function, which converts a sequence to a 32-bit signed integer
(similar to C/C++'s atoi function).

The algorithm for myAtoi(sequence s) is as follows:

Read in and ignore any leading whitespace.
Check if the next character (if not already at the end of the sequence) is '-' or '+'.
Read this character in if it is either. This determines if the final result is negative or positive respectively.
Assume the result is positive if neither is present.
Read in next the characters until the next non-digit character or the end of the input is reached.
The rest of the sequence is ignored.
Convert these digits into an integer (i.e. "123" -> 123, "0032" -> 32). If no digits were read, then the integer is 0.
Change the sign as necessary (from step 2).
If the integer is out of the 32-bit signed integer range [-231, 231 - 1], then clamp the integer so that
it remains in the range. Specifically, integers less than -231 should be clamped to -231,
and integers greater than 231 - 1 should be clamped to 231 - 1.
Return the integer as the final result.

Note:
----
Only the space character ' ' is considered a whitespace character.
Do not ignore any characters other than the leading whitespace or the rest of the sequence after the digits.


Example 1:
Input: s = "42"
Output: 42
Explanation: The underlined characters are what is read in, the caret is the current reader position.
Step 1: "42" (no characters read because there is no leading whitespace)
         ^
Step 2: "42" (no characters read because there is neither a '-' nor '+')
         ^
Step 3: "42" ("42" is read in)
           ^
The parsed integer is 42.
Since 42 is in the range [-231, 231 - 1], the final result is 42.

Example 2:
Input: s = "   -42"
Output: -42
Explanation:
Step 1: "   -42" (leading whitespace is read and ignored)
            ^
Step 2: "   -42" ('-' is read, so the result should be negative)
             ^
Step 3: "   -42" ("42" is read in)
               ^
The parsed integer is -42.
Since -42 is in the range [-231, 231 - 1], the final result is -42.

Example 3:
Input: s = "4193 with words"
Output: 4193
Explanation:
Step 1: "4193 with words" (no characters read because there is no leading whitespace)
         ^
Step 2: "4193 with words" (no characters read because there is neither a '-' nor '+')
         ^
Step 3: "4193 with words" ("4193" is read in; reading stops because the next character is a non-digit)
             ^
The parsed integer is 4193.
Since 4193 is in the range [-231, 231 - 1], the final result is 4193.


Constraints:
0 <= s.length <= 200
s consists of English letters (lower-case and upper-case), digits (0-9), ' ', '+', '-', and '.'.

"""


def my_atoi(sequence: str) -> int:
    # remove all leading spaces
    cleaned = sequence.lstrip()
    if not cleaned:
        return 0

    # read the sign
    sign = 1
    if cleaned[0] in "+-":
        if cleaned[0] == "-":
            sign = -1
        cleaned = cleaned[1:]  # remove the sign

    # read all digits until a non-digit is found
    # and accumulate the total value in the answer variable
    # effectively converting the digits to integer value
    answer = 0
    for char in cleaned:
        if not char.isdigit():
            break
        answer = answer * 10 + int(char)

    # apply the sign, if negative
    answer *= sign

    # check the limits
    low = -(2**31)
    high = 2**31 - 1
    if answer < low:
        return low
    if answer > high:
        return high
    return answer


if __name__ == "__main__":
    inputs = ["42", "   -42", "4193 with words", "", "+-12", "00000-42a1234"]

    for s in inputs:
        v = my_atoi(s)
        print(f"{type(my_atoi(s))}: {v}")
