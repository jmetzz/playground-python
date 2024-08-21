import timeit
from collections import Counter


def custom_sort_string_1(order: str, s: str) -> str:
    char_freq_map = Counter(s)
    used_chars = set()
    answer, not_in_order = "", ""
    for ch in order:
        substr = ch * char_freq_map[ch]
        used_chars.add(ch)
        answer += substr
    for ch in char_freq_map.keys() - used_chars:
        not_in_order += ch * char_freq_map[ch]
    return answer + not_in_order


def custom_sort_string_2(order: str, s: str) -> str:
    char_freq_map = Counter(s)
    answer = ""
    for ch in order:
        freq = char_freq_map.pop(ch, 0)
        if freq:
            answer += ch * freq
    # add the chards not in the custom order
    for ch, freq in char_freq_map.items():
        answer += ch * freq
    return answer


def custom_sort_string_3(order: str, s: str) -> str:
    char_freq_map = Counter(s)
    answer = []
    for ch in order:
        freq = char_freq_map.pop(ch, 0)
        if freq:
            answer.append(ch * freq)
    # add the chards not in the custom order
    for ch, freq in char_freq_map.items():
        answer.append(ch * freq)
    return "".join(answer)


def measure_perf(target_func: str, order: str, s: str, num_runs: int = 100) -> float:
    """Measures the performance of a custom sort function.

    Args:
    ----
        func_name (str): The name of the function to measure.
        order (str): The 'order' argument to pass to the sorting function.
        s (str): The 's' (string) argument to pass to the sorting function.
        number (int, optional): The number of times to run the function. Defaults to 10000.

    Returns:
    -------
        float: The average execution time in seconds.

    """
    # Prepare the namespace with the function and arguments to be accessible by timeit
    namespace = globals().copy()
    namespace.update(locals())

    # Construct the command to execute
    stmt = f"{target_func}('{order}', '{s}')"

    # Measure the execution time
    return timeit.timeit(stmt, globals=namespace, number=num_runs)


if __name__ == "__main__":
    order_str = "uoieaczbf"
    input_str = "zabcdtwporldsswkeiifqaaepomljduiwjithsd"
    num_runs = 1_000_000
    for suffix in range(1, 4):
        func_name = f"custom_sort_string_{suffix}"
        exec_time = measure_perf(func_name, order_str, input_str, num_runs=num_runs)
        print(f"{func_name}' took " f"{exec_time:.5f} seconds to run {num_runs}")
