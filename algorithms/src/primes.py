import math


def is_prime(x):
    if x < 2:
        return False
    for i in range(2, int(math.sqrt(x)) + 1):
        if x % i == 0:
            return False
    return True


def check_prime(number):
    sqrt_number = math.sqrt(number)
    for i in range(2, int(sqrt_number) + 1):
        if (number / i).is_integer():
            return False
    return True


if __name__ == "__main__":
    print(f"check_prime(10,000,000) = {check_prime(10_000_000)}")
    # check_prime(10,000,000) = False
    print(f"check_prime(10,000,019) = {check_prime(10_000_019)}")
    # check_prime(10,000,019) = True
