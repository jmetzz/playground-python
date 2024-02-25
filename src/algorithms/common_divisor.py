"""
The greatest common divisor (GCD) of two integers, also known as the greatest common factor (GCF) or
highest common factor (HCF), is the largest positive integer that divides both numbers without leaving a remainder.
It's a fundamental concept in number theory because it helps in simplifying fractions,
computing least common multiples, and solving various mathematical problems.

Here's how to understand it:

1. **Divisibility**: An integer 'a' is divisible by another integer 'b' (not equal to zero),
if there exists an integer 'k' such that 'a = kb'.

2. **GCD Definition**: The GCD of two integers 'a' and 'b' (not both zero) is the largest
integer 'd' that divides both 'a' and 'b' without leaving a remainder.

3. **Examples**:
   - The GCD of '8' and '12' is '4', since '4' is the largest number that divides
   both '8' and '12' without a remainder.
   - The GCD of '7' and '13' is '1', since '7' and '13' are prime to each other
   (they have no common divisors other than '1).

4. **Calculation Methods**:
   - **Euclidean Algorithm**: A popular and efficient method for finding the GCD of two numbers.
   It is based on the principle that the GCD of two numbers also divides their difference.
   The algorithm involves repeated division steps until a remainder of zero is found.
   The non-zero remainder just before this step is the GCD.
   - **Prime Factorization**: This method involves finding the prime factors of both numbers
   and then multiplying the common factors. However, this method is less efficient for large
   numbers compared to the Euclidean algorithm.

5. **Properties**:
   - The GCD of 'a' and 'b' is the same as the GCD of 'b' and 'a'.
   - The GCD of 'a' and '0' is '|a|', since every number divides '0', and the largest
   number that divides 'a' is '|a|' itself.

"""


def gcd_iterative(a: int, b: int, verbose: bool = False) -> int:
    """Euclidean algorithm method for finding the greatest common divisor (GCD) of two integers.

    It's based on a simple, yet powerful principle from number theory.
    The algorithm can be understood and applied through a series of steps or iterations,
    reducing the problem size at each step until the GCD is found. Here's a breakdown of how it works:

    The key principle behind the Euclidean algorithm is that the GCD of two numbers also divides
    their difference. More formally, for any two positive integers
    'a' and 'b', where 'a > b', the GCD of 'a' and 'b' is the same as the GCD of 'b' and 'a - b'.

    Steps:

    1. **Starting Point**: Begin with two integers, 'a' and 'b', where 'a > b'.
    If 'a < b', you can swap them so that 'a' is always the larger.

    2. **Division Step**: Divide 'a' by 'b' and find the remainder.
    Formally, you calculate 'a mod b', where 'mod' denotes the modulo operation,
    which gives the remainder after division.

    3. **Update Step**: Replace 'a' with 'b' and 'b' with the remainder from the division step.

    4. **Repeat**: Continue the process (division and update steps) until 'b' becomes '0'.
    The non-zero remainder just before 'b' becomes '0' is the GCD of the original pair of integers.

    5. **Conclusion**: When you reach a remainder of '0', the current value of 'a' is
    the GCD of the original two numbers.

    Illustration: finding the GCD of '252' and '198':

    - **Step 1**: Divide '252' by '198', remainder is '54'. So, update 'a = 198', 'b = 54'.
    - **Step 2**: Divide '198' by '54', remainder is '36'. Update 'a = 54', 'b = 36'.
    - **Step 3**: Divide '54' by '36', remainder is '18'. Update 'a = 36', 'b = 18'.
    - **Step 4**: Divide '36' by '18', remainder is '0'. Stop here, as 'b' has become '0'.

    The GCD of '252' and '198' is '18', which is the last non-zero remainder.

    ### Efficiency and Application

    The Euclidean algorithm is highly efficient for computing the GCD of two integers,
    significantly more so than methods involving prime factorization, especially as the numbers involved grow larger.
    Its simplicity and efficiency make it a cornerstone in many areas of mathematics and computer science,
    particularly in algorithms that require simplification of fractions, modular arithmetic,
    and cryptographic algorithms.

    """

    while b != 0:
        if verbose:
            reminder = a % b
            print(f"({a}, {b}) : {reminder} -> {a}")
        a, b = b, a % b
    return a


def gcd_recursive(a: int, b: int) -> int:
    return a if b == 0 else gcd_recursive(b, a % b)


if __name__ == "__main__":
    print(gcd_recursive(253, 198))  # Output will be 11
    print(gcd_iterative(198, 253))  # Output will be 11
