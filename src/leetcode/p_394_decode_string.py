"""394. Decode String
Medium

Given an encoded string, return its decoded string.

The encoding rule is: k[encoded_string], where the encoded_string inside the square brackets
is being repeated exactly k times. Note that k is guaranteed to be a positive integer.

You may assume that the input string is always valid; there are no extra white spaces,
square brackets are well-formed, etc. Furthermore, you may assume that the original data
does not contain any digits and that digits are only for those repeat numbers, k.
For example, there will not be input like 3a or 2[4].

The test cases are generated so that the length of the output will never exceed 105.



Example 1:
Input: s = "3[a]2[bc]"
Output: "aaabcbc"

Example 2:
Input: s = "3[a2[c]]"
Output: "accaccacc"

Example 3:
Input: s = "2[abc]3[cd]ef"
Output: "abcabccdcdcdef"


Constraints:

1 <= s.length <= 30
s consists of lowercase English letters, digits, and square brackets '[]'.
s is guaranteed to be a valid input.
All the integers in s are in the range [1, 300].
"""


def decode_string_non_nested(s: str) -> str:
    # 3[a]2[bc]
    # 3[a | 2[bc

    # 3[a
    # index = 1 --> num = int("3")
    # answer = p[2:] * num = "a" * 3 = aaa

    # 2[bc
    # index = 1 --> num = int("2")
    # answer += p[2:] * num = "bc" * 2 = bcbc = aaabcbc

    answer = ""
    parts = s.split("]")
    for p in parts:
        if p == "":
            continue
        index = p.find("[")
        num = int(p[:index]) if index > 0 else 1
        answer += p[index + 1 :] * num
    return answer


def decode_string_nested(s: str) -> str:
    stack = []
    for e in s:
        if e != "]":
            stack.append(e)
        else:
            # build the pattern
            pattern = ""
            while stack and not stack[-1].isdigit():
                pattern = stack.pop() + pattern
            if pattern[0] == "[":
                pattern = pattern[1:]

            # find out how many times we need to repeat the pattern
            qtt = ""
            while stack and stack[-1].isdigit():
                qtt = stack.pop() + qtt
            qtt = int(qtt) if qtt else 1

            stack.append(pattern * qtt)
    return "".join(stack)


if __name__ == "__main__":
    inputs = [
        ("3[a]2[bc]", "aaabcbc"),
        ("2[abc]3[cd]ef", "abcabccdcdcdef"),
        ("10[leetcode]", "leetcodeleetcodeleetcodeleetcodeleetcodeleetcodeleetcodeleetcodeleetcodeleetcode"),
    ]
    for word in inputs:
        print(decode_string_non_nested(word))
        print()

    inputs = [
        ("3[a]2[bc]", "aaabcbc"),
        ("2[abc]3[cd]ef", "abcabccdcdcdef"),
        ("10[leetcode]", "leetcodeleetcodeleetcodeleetcodeleetcodeleetcodeleetcodeleetcodeleetcodeleetcode"),
        ("3[a2[c]]", "accaccacc"),
        ("2[abc]3[c5[d]]ef", "abcabccdddddcdddddcdddddef"),
        ("3[z]2[2[y]pq4[2[jk]e1[f]]]ef", "zzzyypqjkjkefjkjkefjkjkefjkjkefyypqjkjkefjkjkefjkjkefjkjkefef"),
    ]
    for word in inputs:
        print(decode_string_nested(word))
        print()
