"""
150. Evaluate Reverse Polish Notation
Medium
You are given an array of strings tokens that represents an arithmetic expression
in a Reverse Polish Notation.

Evaluate the expression. Return an integer that represents the value of the expression.

Note that:
The valid operators are '+', '-', '*', and '/'.
Each operand may be an integer or another expression.
The division between two integers always truncates toward zero.
There will not be any division by zero.
The input represents a valid arithmetic expression in a reverse polish notation.
The answer and all the intermediate calculations can be represented in a 32-bit integer.


Example 1:
Input: tokens = ["2","1","+","3","*"]
Output: 9
Explanation: ((2 + 1) * 3) = 9

Example 2:
Input: tokens = ["4","13","5","/","+"]
Output: 6
Explanation: (4 + (13 / 5)) = 6

Example 3:
Input: tokens = ["10","6","9","3","+","-11","*","/","*","17","+","5","+"]
Output: 22
Explanation: ((10 * (6 / ((9 + 3) * -11))) + 17) + 5
= ((10 * (6 / (12 * -11))) + 17) + 5
= ((10 * (6 / -132)) + 17) + 5
= ((10 * 0) + 17) + 5
= (0 + 17) + 5
= 17 + 5
= 22


Constraints:

1 <= tokens.length <= 104
tokens[i] is either an operator: "+", "-", "*", or "/", or an integer in the range [-200, 200].
"""
import math
import operator
from typing import List


def _my_div(a, b):
    value = a / b
    if -1 < value < 1:
        value = math.floor(value) if value > 0 else math.ceil(value)
    return value


class EvaluateReversePolishNotation:
    OPERATIONS = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": _my_div}

    def solve(self, tokens: List[str]) -> int:
        stack = []
        for token in tokens:
            if token in self.OPERATIONS:
                op2 = stack.pop()
                op1 = stack.pop()
                value = self.OPERATIONS[token](op1, op2)
                stack.append(int(value))
            else:
                stack.append(int(token))
        return stack.pop()
