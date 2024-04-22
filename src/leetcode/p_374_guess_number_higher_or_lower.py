"""
https://leetcode.com/problems/guess-number-higher-or-lower/description

374. Guess Number Higher or Lower
Easy

We are playing the Guess Game. The game is as follows:

I pick a number from 1 to n. You have to guess which number I picked.

Every time you guess wrong, I will tell you whether the number I picked is higher or lower than your guess.

You call a pre-defined API int guess(int num), which returns three possible results:

-1: Your guess is higher than the number I picked (i.e. num > pick).
1: Your guess is lower than the number I picked (i.e. num < pick).
0: your guess is equal to the number I picked (i.e. num == pick).
Return the number that I picked.



Example 1:

Input: n = 10, pick = 6
Output: 6
Example 2:

Input: n = 1, pick = 1
Output: 1
Example 3:

Input: n = 2, pick = 1
Output: 1


Constraints:

1 <= n <= 231 - 1
1 <= pick <= n
"""


class GuessNumber:
    def __init__(self, pick) -> None:
        self._pick = pick

    def _guess(self, guess: int):
        """
        Returns:
            -1: Your guess is higher than the number I picked (i.e. num > pick).
            1: Your guess is lower than the number I picked (i.e. num < pick).
            0: your guess is equal to the number I picked (i.e. num == pick).
        """
        if guess == self._pick:
            return 0
        elif guess > self._pick:
            return -1
        else:
            return 1

    def _bin_search(self, lo, hi):
        if lo > hi:
            # Base case: not found (should not happen in a correct scenario)
            return -1

        target = lo + (hi - lo) // 2

        match self._guess(target):
            case 0:
                return target
            case -1:
                return self._bin_search(lo, target - 1)
            case 1:
                return self._bin_search(target + 1, hi)

    def guess_number_recursive(self, n: int) -> int:
        return self._bin_search(1, n)

    def guess_number_iterative(self, n: int) -> int:
        lo, hi = 1, n
        while lo <= hi:
            target = lo + (hi - lo) // 2

            match self._guess(target):
                case 0:
                    return target
                case -1:
                    hi = target - 1
                case 1:
                    lo = target + 1
        return -1


if __name__ == "__main__":
    print(GuessNumber(6).guess_number_recursive(10))
    print(GuessNumber(1).guess_number_recursive(1))
    print(GuessNumber(1).guess_number_recursive(2))

    print(GuessNumber(6).guess_number_iterative(10))
    print(GuessNumber(1).guess_number_iterative(1))
    print(GuessNumber(1).guess_number_iterative(2))
