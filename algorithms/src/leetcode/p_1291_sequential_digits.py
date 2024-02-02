from typing import List


class SequentialDigits:
    def solve(self, low: int, high: int) -> List[int]:
        digits = "123456789"
        n = len(digits)
        max_len = len(str(high))
        min_len = len(str(low))
        answer = []

        for size in range(min_len, max_len + 1):
            for i in range(n):
                if i + size <= n:
                    value = int(digits[i: i + size])
                    if low <= value <= high:
                        answer.append(value)

        return answer


if __name__ == '__main__':
    print(SequentialDigits().solve(100, 300))
    print(SequentialDigits().solve(1000, 13000))
