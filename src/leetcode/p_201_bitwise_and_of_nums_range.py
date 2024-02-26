def range_bitwise_and__solve_naive_loop(left: int, right: int) -> int:
    if left == 0:
        return 0
    answer = left
    for e in range(left + 1, right + 1):
        answer &= e
    return answer


def range_bitwise_and__solve_bitwise(left: int, right: int) -> int:
    cnt = 0
    while left != right:
        left >>= 1
        right >>= 1
        cnt += 1
    return left << cnt


def range_bitwise_and__solve_kernighan(left: int, right: int) -> int:
    """
    The main idea behind this algorithm is that when we
    subtract one from any number, it inverts all the bits
    after the rightmost set bit i.e. it turns 1 to 0 and 0 to 1.

    Then, decrement 1 from right until right reaches left - 1.
    """
    while right > left:
        right &= right - 1
    return right & left


if __name__ == "__main__":
    print(range_bitwise_and__solve_naive_loop(5, 7))  # 4
    print(range_bitwise_and__solve_naive_loop(6, 7))  # 6
    print(range_bitwise_and__solve_naive_loop(0, 0))  # 0
    print(range_bitwise_and__solve_naive_loop(1, 4))  # 0
    print(range_bitwise_and__solve_naive_loop(6, 9))  # 0
    print("===")
    print(range_bitwise_and__solve_kernighan(5, 7))  # 4
    print(range_bitwise_and__solve_kernighan(6, 7))  # 6
    print(range_bitwise_and__solve_kernighan(0, 0))  # 0
    print(range_bitwise_and__solve_kernighan(1, 4))  # 0
    print(range_bitwise_and__solve_kernighan(6, 9))  # 0
    print(range_bitwise_and__solve_kernighan(5664, 2**31 - 1))  # 0
