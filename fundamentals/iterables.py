#!/usr/bin/env python3


from math import factorial, sqrt
from itertools import islice, count


def size_of_factorial(r):
    return [len(str(factorial(x))) for x in range(r)]
    
def is_prime(x):
    if x < 2:
        return False
    for i in range(2, int(sqrt(x)) + 1):
        if x % i == 0:
            return False
    return True


# [x for x in range(101) if is_prime(x)]
# prime_square_divisions = {x*x: (1, x, x*x) for x in range(101) if is_prime(x)}

thousand_primes = islice((x for x in count() if is_prime(x)), 1000)
print(list(thousand_primes))
print(sum(islice((x for x in count() if is_prime(x)), 1000)))



