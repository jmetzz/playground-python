from collections import deque


class MovingAverage:
    def __init__(self, size: int):
        self._elements = deque(maxlen=size)
        self._current_sum = 0.0

    def next(self, val: int, verbose=False) -> float:
        if len(self._elements) >= self._elements.maxlen:
            p = self._elements.pop()
            self._current_sum -= p

        self._elements.appendleft(val)
        self._current_sum += val

        if verbose:
            print(f"Current sequence: {self._elements}")
        return self._current_sum / len(self._elements)

    def reset(self):
        self._elements.clear()
        self._current_sum = 0.0


def warm_up(avg_wrapper, n, verbose=False):
    for i in range(n):
        avg_wrapper.next(i, verbose)


if __name__ == "__main__":
    size, next_value = 10, 15
    avg_sequence = MovingAverage(size)

    for v in range(next_value + 1):
        print(avg_sequence.next(v, True))

    print("Resetting the MovingAverage object")
    avg_sequence.reset()
    print("--- Warm_up ---")
    warm_up(avg_sequence, next_value, True)
    print("Simulate streaming:")
    for v in range(1, 20):
        print(avg_sequence.next(v, True))
