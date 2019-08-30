#!/usr/bin/env python3


from itertools import islice, count
from math import factorial, sqrt

from algorithms.primes import is_prime


def size_of_factorial(r):
    return [len(str(factorial(x))) for x in range(r)]


# [x for x in range(101) if is_prime(x)]
# prime_square_divisions = {x*x: (1, x, x*x) for x in range(101) if is_prime(x)}

thousand_primes = islice((x for x in count() if is_prime(x)), 1000)
print(list(thousand_primes))
print(sum(islice((x for x in count() if is_prime(x)), 1000)))
