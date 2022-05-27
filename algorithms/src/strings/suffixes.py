from typing import Any, List


class Alphabet:
    pass
    # def __init__(self, elements):


class Trie:
    """
    A multi-way tree structure useful for storing strings over an alphabet.

    Trie describes a tree data structure suitable for use as an associative array,
    where branches or edges correspond to parts of a key.

    Each of its nodes will have a number of sons equal to the size of the alphabet used
    by the strings that are needed to be stored.
    Every edge going from the father toward its sons is labeled with a different letter
    of the alphabet. The labels on a path starting from the root and ending in a leaf
    will form a string stored in that tree.

    All strings sharing a common stem or prefix hang off a common node.
    When the strings are words over {a..z}, a node has at most 27 children,
    one for each letter plus a terminator.

    """

    def __init__(self):
        pass

    def all_prefixes(self):
        """
        All strings in the trie can be recovered by
        a depth-first scan of the tree.
        """


class RadixTrie:
    """
    A compressed representation of a trie.

    Describe a form of trie that condenses common prefix parts, resulting in
    less memory requirement and less node accesses.
    """


class PatriciaTree:
    """
    A compact representation of a trie.

    Also known as radix tree [https://xlinux.nist.gov/dads/HTML/patriciatree.html]
    Actually, Patricia tries are radix trees with radix equal to 2.

    Patricia stands for
        Practical Algorithm To Retrieve Information Coded in Alphanumeric
    from [original paper by Donald R. Morrison]
    (https://dl.acm.org/citation.cfm?id=321481).

    Any node that is an only child is merged with its parent.

    Extra resource:
    https://stackoverflow.com/questions/14708134/what-is-the-difference-between-trie-and-radix-trie-data-structures

    """


class SiString:
    def __init__(self, text):
        pass

    def compare(self, other):
        pass


class PATTree:
    """
    A PAT tree is a Patricia tree constructed over all the possible
    sistrings of a text.

    A Patricia tree (Morrison 1968; Knuth 1973; Flajolet and Sedgewick 1986;
    and Gonnet 1988) is a digital tree where the individual bits of the keys are used
    to decide on the branching.

    A zero bit will cause a branch to the left subtree, a one bit will cause a
    branch to the right subtree. Hence Patricia trees are binary digital trees.
    In addition, Patricia trees have in each internal node an indication of which
    bit of the query is to be used for branching. This may be given by an absolute
    bit position, or by a count of the number of bits to skip.

    This allows internal nodes with single descendants to be eliminated,
    and thus all internal nodes of the tree produce a useful branching,
    that is, both subtrees are non-null. Patricia trees are very similar
    to compact suffix trees or compact position trees (Aho et al. 1974).

    By using the PAT tree, one can search for a pattern query in a binary
    search fashion. Once we reach the desired node we have to make one
    final comparison with one of the sistrings stored in an external node of
    the current subtree, to ensure that all the skipped bits coincide.
    If they do not coincide, then the key is not in the tree.

    Important notes:
    - store key values at external nodes
    - internal nodes have no key information, just the skip counter and the pointers
     to the subtrees
    - leaf/external nodes are references to sistrings
    - bucketing of external nodes is used to decrease memory consumption.
    The external nodes inside a bucket do not have any structure associated
    with them, and any search that has to be done in the bucket has to be done
    on all the members of the bucket

    References:
    MORRISON, D. 1968. "PATRICIA-Practical Algorithm to Retrieve Information
    Coded in Alphanumeric." JACM, 15; 514-34.
    KNUTH, D. 1973. The Art of Computer Programming: Sorting and Searching,
    vol. 3. Reading, Mass.: Addison-Wesley.
    FLAJOLET, P. and R. SEDGEWICK. 1986. "Digital Search Trees Revisited."
    SIAM J Computing, 15; 748-67.
    GONNET, G. 1988. "Efficient Searching of Text and Pictures (extended abstract)."
    Technical Report OED-88-02, Centre for the New OED., University of Waterloo.

    """


class SuffixTree:
    """
    A Trie-like tree that represents the suffixes of a given text.

    A Suffix Tree is a data-structure that allows many problems on strings
    (sequences of characters) to be solved quickly. If txt=t1t2...ti...tn is a string,
    then Ti=titi+1...tn is the suffix of txt that starts at position i.

    The suffix trie is made by compressing all the suffixes of A1…An – 1 into a trie.
    Two or more common prefixes share a common path from the root of the suffix tree
    (as in a PATRICIA tree).

    ex: http://www.allisons.org/ll/AlgDS/Tree/Suffix/
    """

    def _lcp(self, text, suffix_array):
        return [self._common_prefix_length(text, i, i + 1) for i in range(suffix_array)]

    @staticmethod
    def _common_prefix_length(text: str, i: int, j: int) -> int:
        count = 0
        while text[i + count] == text[j + count]:
            count += 1
        return count


class SuffixArray:
    """
    store only the order of the suffixes, not the suffixes themselves.

    1- create the cyclic shifts
    2- order the cyclic shifts
    3- remove all characters after the marker in each cyclic shift

    https://web.stanford.edu/class/cs97si/suffix-array.pdf
    """

    def __init__(self, alphabet: List[Any], mark="$"):
        self.alphabet = [mark] + alphabet
        self.alphabet_size = len(self.alphabet)
        self.alphabet_index = dict(zip(self.alphabet, range(len(self.alphabet))))

    def build(self, text: str):
        """
        Builds the suffix array from the input string.

        Assumes last character is $.

        Complexity time: (|text| log|text| + |alphabet|)
        :param text:
        :return:
        """
        order = self.counting_sort_characters(text)
        classes = self.compute_character_classes(text, order)
        shift_length = 1
        while shift_length < len(text):
            order = self.stable_sort_cyclic_shifts(text, shift_length, order, classes)
            classes = self.update_classes(order, classes, shift_length)
            shift_length *= 2
        return order

    def counting_sort_characters(self, text: str) -> List[int]:
        """
        Uses counting sort to sort the characters in the input text.

        Complexity time: (|text| + |alphabet|)
        :param text:
        :return:

        """
        input_size = len(text)
        order = [0] * input_size
        count = [0] * self.alphabet_size

        # count the number of each chat in the string
        for idx in range(input_size):
            count[self.alphabet_index[text[idx]]] += 1
        # computes the partial sums of the count array
        for j in range(1, self.alphabet_size):
            count[j] += count[j - 1]

        # from right to left, build up the order
        for idx in range(input_size - 1, -1, -1):
            character = text[idx]
            count[self.alphabet_index[character]] -= 1
            order[count[self.alphabet_index[character]]] = idx
        return order

    @staticmethod
    def compute_character_classes(text: str, order: List[int]) -> List[int]:
        """
        This algorithm run with complexity time: O(|text|)
        :param text:
        :param order:
        :return:
        """
        classes = [0] * len(text)
        classes[order[0]] = 0
        for idx in range(1, len(text)):
            previous_class = classes[order[idx - 1]]
            if text[order[idx]] == text[order[idx - 1]]:
                classes[order[idx]] = previous_class
            else:
                classes[order[idx]] = previous_class + 1

        return classes

    @staticmethod
    def stable_sort_cyclic_shifts(
        text: str, shift_length, order: List[int], classes: List[int]
    ) -> List[int]:
        """
        Uses counting sort to stable sort the cyclic shifts with the given length.

        O(|text|) time complexity
        :param text:
        :param shift_length:
        :param order:
        :param classes:
        :return:
        """
        input_size = len(text)
        count = [0] * input_size
        for idx in range(input_size):
            count[classes[idx]] = count[classes[idx]] + 1

        for j in range(1, input_size):
            count[j] = count[j] + count[j - 1]

        new_order = [0] * input_size
        # reverse order loop guarantees the sorting stability
        for idx in range(input_size - 1, -1, -1):
            # circle indexing
            start = (order[idx] - shift_length + input_size) % input_size
            cl = classes[start]
            count[cl] -= 1
            new_order[count[cl]] = start

        return new_order

    @staticmethod
    def update_classes(
        new_order: List[int], classes: List[int], shift_length: int
    ) -> List[int]:
        """
        Complexity time is linear.

        :param new_order:
        :param classes:
        :param shift_length:
        :return:
        """
        n = len(new_order)
        new_classes = [0] * n

        for idx in range(1, n):
            current = new_order[idx]
            second_half_start = current + shift_length
            previous = new_order[idx - 1]
            previous_second_half_start = (previous + shift_length) % n
            if (
                classes[current] != classes[previous]
                or classes[second_half_start] != classes[previous_second_half_start]
            ):
                new_classes[current] = new_classes[previous] + 1
            else:
                new_classes[current] = new_classes[previous]
        return new_classes


class BurrowsWheelerTransform:
    pass
    # http://www.allisons.org/ll/AlgDS/Strings/BWT/


if __name__ == "__main__":
    input_text = "ababaa$"
    size = len(input_text)
    sa = SuffixArray(["a", "b"])
    indexes = sa.build(input_text)
    for i in indexes:
        print(input_text[i: size - 1])

    print("----")
    input_text = "ababaacabc$"
    size = len(input_text)
    sa = SuffixArray(["a", "b", "c"])
    indexes = sa.build(input_text)
    for i in indexes:
        print(input_text[i: size - 1])
