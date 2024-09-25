import math


def is_prime(x):
    if x < 2:
        return False
    return all(x % i != 0 for i in range(2, int(math.sqrt(x)) + 1))


def check_prime(number):
    sqrt_number = math.sqrt(number)
    return all(not (number / i).is_integer() for i in range(2, int(sqrt_number) + 1))


def prime_factors(n: int) -> list[int]:
    factors = []
    # Check for divisibility by 2
    while n % 2 == 0:
        factors.append(2)
        n = n // 2

    # Incrementing by 2 for Odd Factors:
    # Start from 3 and increment by 2 (since even numbers are not prime except 2)
    # This approach avoids checking even numbers (other than 2) which cannot be prime factors,
    # effectively halving the number of divisions needed.
    # Moreover, it checks potential factors up to the square root of the current number (eg 'int(n ** 0.5) + 1').
    # This is because, for any factor larger than the square root, there must be a corresponding factor
    # smaller than the square root, so there's no need to check beyond it
    for i in range(3, int(n**0.5) + 1, 2):
        while n % i == 0:
            factors.append(i)
            n = n // i

    # If n becomes a prime number > 2
    if n > 2:
        factors.append(n)

    return factors


if __name__ == "__main__":
    print(f"check_prime(10,000,000) = {check_prime(10_000_000)}")
    # check_prime(10,000,000) = False
    print(f"check_prime(10,000,019) = {check_prime(10_000_019)}")
    # check_prime(10,000,019) = True

    print(prime_factors(1))
    print(prime_factors(2))
    print(prime_factors(5))
    print(prime_factors(25))
    print(prime_factors(88))
