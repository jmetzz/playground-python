def is_subsequence(s: str, t: str) -> bool:
    s_start, t_start = 0, 0
    while s_start < len(s) and t_start < len(t):
        # does the curr char exist in the remaining string
        idx = t[t_start:].find(s[s_start])
        if idx == -1:
            return False
        s_start += 1
        t_start += idx + 1
    return s_start == len(s)


def is_subsequence_2(s: str, t: str) -> bool:
    suffix_idx, vocab_idx = 0, 0
    suffix_size = len(s)
    while suffix_idx < suffix_size and vocab_idx < len(t):
        if s[suffix_idx] == t[vocab_idx]:
            suffix_idx += 1
        vocab_idx += 1
    return suffix_idx == suffix_size


if __name__ == "__main__":
    print(is_subsequence("abc", "ahbgdc"))  # True
    print(is_subsequence("axc", "ahbgdc"))  # False
    print(is_subsequence("acb", "ahbgdc"))  # False
    print(is_subsequence("aaaaaa", "bbaaaa"))  # False
