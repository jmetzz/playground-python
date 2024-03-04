import random
from typing import Set, Tuple

WORDS = [
    "stopping",
    "starting",
    "continue",
]


def fetch_random_word(hint_size: int = 1) -> Tuple[str, str]:
    word = random.choice(WORDS)
    keep_chars = random.sample(word, hint_size)
    hint = str(word)
    for ch in word:
        if ch in keep_chars:
            continue
        hint = hint.replace(ch, "-")
    return word, hint


def guess(word: str, reveled: Set[str], ch: str) -> Tuple[Set[str], str]:
    if ch in word:
        return reveled.union(ch), "Good job!"
    else:
        return reveled, "Fail. Try again!"


if __name__ == "__main__":
    w, h = fetch_random_word(hint_size=2)
    print(f"Guess this word: {w}")
    print(f"Hint: {h}")
    print()

    print("Guessing 'a'")
    r, msg = guess(
        w,
        set([c for c in h if c != "-"]),
        "a",
    )
    print(f"{msg} --> reveled chars: {r}")

    print("Guessing 'g'")
    r, msg = guess(
        w,
        r,
        "g",
    )
    print(f"{msg} --> reveled chars: {r}")
