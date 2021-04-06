from typing import List


def counting_sort(elements: List[int]) -> None:
    # find the max element
    # Example: [4, 2, 2, 8, 3, 3, 1]; max = 8
    max = 0
    size = len(elements)
    for index in range(0, size):
        if elements[index] > elements[max]:
            max = index

    # initialize and 0-array of size max + 1, which is used to store
    # the count of the elements present in the input array
    # at it's respective index in the count array.
    # Example: [4, 2, 2, 8, 3, 3, 1] -> [0, 1, 2, 2, 1, 0, 0, 0, 1]
    count = [0] * (elements[max] + 1)
    for index in range(0, size):
        count[elements[index]] += 1

    # Store cumulative sum of the elements of the count array.
    # It helps in placing the elements into the correct index.
    # If there are x elements less than y, its position
    # should be at x-1.
    # For example: In the accumulated sum array [0, 1, 3, 5, 6, 6, 6, 6, 7],
    # the count of 4 is 6. It denotes that there are 5 elements smaller
    # than 4. Thus, the position of 4 in the sorted array is 5th.
    for index in range(1, len(count)):
        count[index] += count[index - 1]

    # Find the index of each element of the original array in count array.
    # This gives the cumulative count.
    # Place the element at the index calculated.
    # After placing each element at its correct position, decrease its count by one.
    # Example:
    # original array: [4, 2, 2, 8, 3, 3, 1]
    #                  *
    #                  |
    #                  |-----------|
    #                              |
    #                              *
    #           count [0, 1, 3, 5, 6, 6, 6, 6, 7]
    #                              *
    #                              |
    #                              ----
    #                                  |
    #                                  *
    #           output [0, 0, 0, 0, 0, 4, 0]
    #    updated count [0, 1, 3, 5, 5, 6, 6, 6, 7]
    #
    # original array: [4, 2, 2, 8, 3, 3, 1]
    #                     *
    #                     |
    #                     |--|
    #                        |
    #                        *
    #           count [0, 1, 3, 5, 5, 6, 6, 6, 7]
    #                        *
    #                        |
    #                        --
    #                         |
    #                         *
    #           output [0, 0, 2, 0, 0, 4, 0]
    #    updated count [0, 1, 2, 5, 5, 6, 6, 6, 7]

    output = [0] * size
    for index in range(0, size):
        e = elements[index]
        element_count = count[e]
        output[element_count - 1] = e
        count[e] -= 1

    # overwrite the input array
    for i in range(0, size):
        elements[i] = output[i]


if __name__ == "__main__":
    data = [4, 2, 2, 8, 3, 3, 1]
    counting_sort(data)
    print("Sorted Array in Ascending Order: ")
    print(data)
