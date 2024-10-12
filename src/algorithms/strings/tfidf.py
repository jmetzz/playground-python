import itertools
import math
from collections import Counter, defaultdict
from collections.abc import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def build_indices(corpus: Iterable[str]) -> tuple[dict[set], dict[int, Counter]]:
    """Builds term and inverted indices from a given corpus of documents.

    This function processes a collection of documents (strings) and creates two indices:
    a term index that maps each term to the set of document indices containing that term,
    and an inverted index that maps each document index to a dictionary of term frequencies.

    Args:
        corpus (Iterable[str]): An iterable containing documents as strings. Each document is
        expected to be a single string representing a collection of terms.

    Returns:
        tuple[dict, dict]: A tuple containing two dictionaries:
            - term_index (defaultdict(set)): A mapping of terms to a set of document indices
              that contain each term.
            - inverted_index (defaultdict(dict)): A mapping of document indices to a
              dictionary where keys are terms and values are their respective frequencies
              in the document.

    Example:
        >>> corpus = ["this is a document", "this document is another document"]
        >>> term_index, inverted_index = build_indices(corpus)
        >>> print(term_index)
        defaultdict(<class 'set'>, {'this': {0, 1}, 'is': {0, 1}, 'a': {0}, 'document': {0, 1}, 'another': {1}})
        >>> print(inverted_index)
        defaultdict(<class 'dict'>, {0: Counter({'document': 1, 'this': 1, 'is': 1, 'a': 1}),
            1: Counter({'document': 2, 'this': 1, 'is': 1, 'another': 1})})
    """
    term_index = defaultdict(set)  # term : {doc_idx*} --> which docs contain the term
    inverted_index = defaultdict(dict)  # doc_idx : {term : freq}
    for i, document in enumerate(corpus):
        counter = Counter(document)
        for word in counter.keys():
            term_index[word].add(i)
        inverted_index[i] = counter
    return term_index, inverted_index


def term_frequency(term: str, doc_idx: int, doc_size: int, inverted_index: dict[int, dict[str, int]]) -> float:
    """Measures how frequently a term appears in a document.

    The term frequency is calculated as the number of times a term appears in a document
    divided by the total number of terms in that document.

    Args:
        term (str): The target term to calculate the frequency for.
        doc_idx (int): The target document index to calculate the frequency from.
        doc_size: The number of tokens in the target document.
        inverted_index (dict[int, dict[str, int]]): A mapping of each document index to a dictionary
            of term frequencies.

    Returns:
        float: The relative term frequency of the term in the specified document. Returns 0 if the
        document index is invalid or the document is empty.

    Example:
        >>> corpus = [["this", "is", "a", "document"], ["this", "document", "is", "another"]]
        >>> inverted_index = {
        ...     0: {"this": 1, "is": 1, "a": 1, "document": 1},
        ...     1: {"this": 1, "document": 2, "is": 1, "another": 1},
        ... }
        >>> term_frequency("this", 0, inverted_index, corpus)
        0.25
    """
    if doc_idx not in inverted_index or doc_size <= 0:
        return 0
    return inverted_index[doc_idx].get(term, 0) / doc_size


def inverse_document_frequency(term: str, term_index: dict[str, list[int]], corpus_size: int) -> float:
    """Measures how rare or common a word is across all documents in the corpus.

    The inverse document frequency is calculated as the logarithm of the total number of documents
    divided by the number of documents that contain the term. Smoothing is applied to avoid
    division by zero.

    Args:
        term (str): The target term to calculate the inverse frequency for.
        term_index (dict[str, list[int]]): A mapping of each term to a list of document indices
            that contain the term.
        corpus_size (int): The number of documents in the corpus.

    Returns:
        float: The smoothed inverse document frequency of the term.

    Example:
        >>> corpus = [["this", "is", "a", "document"], ["this", "document", "is", "another"]]
        >>> term_index = {"this": [0, 1], "document": [0, 1], "is": [0, 1], "another": [1]}
        >>> inverse_document_frequency("this", term_index, len(corpus))
        0.6309297535714574
    """
    # using +1 to prevent div by zero in case the term is not present in any document.
    docs_containing_term = len(term_index[term])
    return math.log((1 + corpus_size) / (1 + docs_containing_term)) + 1


def compute_tfidf(
    term: str, doc_idx: int, doc_size: int, term_index, inverted_index: dict[int, dict[str, int]], corpus_size: int
) -> float:
    """Calculates the TF-IDF score for a term in a document.

    The TF-IDF score is computed as the product of the term frequency and the inverse document frequency.

    Args:
        term (str): The target term to calculate the TF-IDF score for.
        doc_idx (int): The target document index to calculate the score from.
        doc_size: The number of tokens in the target document.
        term_index (dict[str, list[int]]): A mapping of each term to a list of document indices
            that contain the term.
        inverted_index (dict[int, dict[str, int]]): A mapping of each document index to a
            dictionary of term frequencies.
        corpus_size (int): The number of documents in the corpus.

    Returns:
        float: The TF-IDF score for the term in the specified document.

    Example:
        >>> corpus = [["this", "is", "a", "document"], ["this", "document", "is", "another"]]
        >>> term_index = {"this": [0, 1], "document": [0, 1], "is": [0, 1], "another": [1]}
        >>> inverted_index = {
        ...     0: {"this": 1, "is": 1, "a": 1, "document": 1},
        ...     1: {"this": 1, "document": 2, "is": 1, "another": 1},
        ... }
        >>> compute_tfidf("this", 0, len(corpus[0]), term_index, inverted_index, len(corpus))
        0.25 * idf_value  # Replace `idf_value` with the actual value from inverse_document_frequency
    """

    return term_frequency(term, doc_idx, doc_size, inverted_index) * inverse_document_frequency(
        term, term_index, corpus_size
    )


class TfidfV1:
    def __init__(self) -> None:
        self._vocabulary = []
        self._fitted = False

    def fit(self, corpus: Iterable[str]) -> None:
        """Learn vocabulary and idf from training set."""
        splitted_corpus = [doc.split() for doc in corpus]
        all_terms = itertools.chain(*splitted_corpus)
        self._vocabulary = sorted(list(set(all_terms)))
        self._fitted = True

    def _build_indices(self, tokenized_corpus: Iterable[str]) -> tuple[dict, dict]:
        term_index = defaultdict(set)  # term : [doc_idx*] --> which docs contain the term
        inverted_index = defaultdict(Counter)  # doc_idx : {term : freq}
        for doc_idx, terms in enumerate(tokenized_corpus):
            counter = Counter(terms)
            for term in counter.keys():
                term_index[term].add(doc_idx)
            inverted_index[doc_idx] = counter
        return term_index, inverted_index

    def transform(self, corpus: Iterable[str]) -> np.ndarray:
        """Transform documents to document-term matrix.

        Uses the vocabulary and document frequencies (df) learned by fit (or fit_transform).
        """
        if not self._fitted:
            raise Exception("Model not fitted")

        tokenized_corpus = [doc.split() for doc in corpus]
        term_index, inverted_index = self._build_indices(tokenized_corpus)

        num_docs = len(tokenized_corpus)
        vocab_size = len(self._vocabulary)
        result_matrix = np.zeros((num_docs, vocab_size), dtype=float)  # inefficient full matrix
        for doc_idx in range(num_docs):
            for w_idx, term in enumerate(self._vocabulary):
                if term in tokenized_corpus[doc_idx]:
                    term_freq = (
                        inverted_index[doc_idx].get(term, 0) / len(tokenized_corpus[doc_idx])
                        if len(tokenized_corpus[doc_idx]) > 0
                        else 0
                    )
                    num_docs_containing_term = len(term_index[term])
                    # The use of +1 in the numerator also serves as smoothing.
                    # It prevents the IDF value from becoming too extreme, especially for rare terms
                    idf = math.log((1 + num_docs) / (1 + num_docs_containing_term)) + 1
                    result_matrix[doc_idx][w_idx] = term_freq * idf

        return result_matrix

    def fit_transform(self, corpus: Iterable[str]) -> np.ndarray:
        """Learn vocabulary and idf, return document-term matrix.

        In this version, calling fit_transform is equivalent to fit followed by transform.
        """
        self.fit(corpus)
        return self.transform(corpus)


class TfidfV2:
    def __init__(self) -> None:
        self._vocabulary = []
        self._fitted = False

    def fit(self, corpus: Iterable[str]) -> None:
        """Learn vocabulary and idf from training set."""
        splitted_corpus = [doc.split() for doc in corpus]
        all_terms = itertools.chain(*splitted_corpus)
        self._vocabulary = sorted(list(set(all_terms)))
        self._fitted = True

    def _build_indices(self, tokenized_corpus: Iterable[str]) -> tuple[dict, dict]:
        term_index = defaultdict(set)  # term : [doc_idx*] --> which docs contain the term
        inverted_index = defaultdict(Counter)  # doc_idx : {term : freq}
        for doc_idx, terms in enumerate(tokenized_corpus):
            counter = Counter(terms)
            for term in counter.keys():
                term_index[term].add(doc_idx)
            inverted_index[doc_idx] = counter
        return term_index, inverted_index

    def transform(self, corpus: Iterable[str]) -> np.ndarray:
        """Transform documents to document-term matrix.

        Uses the vocabulary and document frequencies (df) learned by fit (or fit_transform).
        """
        if not self._fitted:
            raise Exception("Model not fitted")

        tokenized_corpus = [doc.split() for doc in corpus]
        term_index, inverted_index = self._build_indices(tokenized_corpus)

        num_docs = len(tokenized_corpus)
        vocab_size = len(self._vocabulary)
        result_matrix = np.zeros((num_docs, vocab_size), dtype=float)  # inefficient full matrix
        for doc_idx in range(num_docs):
            for w_idx, term in enumerate(self._vocabulary):
                if term in tokenized_corpus[doc_idx]:
                    term_freq = (
                        inverted_index[doc_idx].get(term, 0) / len(tokenized_corpus[doc_idx])
                        if len(tokenized_corpus[doc_idx]) > 0
                        else 0
                    )
                    num_docs_containing_term = len(term_index[term])
                    # The use of +1 in the numerator also serves as smoothing.
                    # It prevents the IDF value from becoming too extreme, especially for rare terms
                    idf = math.log((1 + num_docs) / (1 + num_docs_containing_term)) + 1
                    result_matrix[doc_idx][w_idx] = term_freq * idf

        # Apply L2 normalization (similar to sklearn's default behavior)
        norms = np.linalg.norm(result_matrix, axis=1, keepdims=True)
        return np.divide(result_matrix, norms, where=norms != 0)  # Avoid division by zero

    def fit_transform(self, corpus: Iterable[str]) -> np.ndarray:
        """Learn vocabulary and idf, return document-term matrix.

        In this version, calling fit_transform is equivalent to fit followed by transform,
        but this implementation is more efficient.
        """
        tokenized_corpus = [doc.split() for doc in corpus]
        num_docs = len(tokenized_corpus)

        # Building the vocabulary and indices
        term_index, inverted_index = self._build_indices(tokenized_corpus)
        vocab_size = len(term_index.keys())
        self._vocabulary = list(term_index.keys())  # improve perf by not sorting the vocab
        vocabulary_positions = {term: idx for idx, term in enumerate(self._vocabulary)}

        # Initialize the result matrix
        result_matrix = np.zeros((num_docs, vocab_size), dtype=float)  # still inefficient full matrix

        # Precompute IDFs
        num_docs_float = float(num_docs)
        idfs = {}
        for term, doc_indices in term_index.items():
            num_docs_containing_term = len(doc_indices)
            idfs[term] = math.log((1 + num_docs_float) / (1 + num_docs_containing_term)) + 1

        # Fill the result matrix
        for doc_idx in range(num_docs):
            doc_length = len(tokenized_corpus[doc_idx])
            for term, term_freq in inverted_index[doc_idx].items():
                # Calculate term frequency
                term_freq_normalized = term_freq / doc_length if doc_length > 0 else 0
                result_matrix[doc_idx][vocabulary_positions[term]] = term_freq_normalized * idfs[term]

        # Apply L2 normalization (similar to sklearn's default behavior)
        norms = np.linalg.norm(result_matrix, axis=1, keepdims=True)
        return np.divide(result_matrix, norms, where=norms != 0)  # Avoid division by zero


if __name__ == "__main__":
    corpus = ["good boy", "good girl", "good boy girl"]
    tokenized_corpus = [doc.split() for doc in corpus]
    corpus_size = len(tokenized_corpus)
    term_index, inverted_index = build_indices(tokenized_corpus)
    target_term, target_doc = "good", 2
    target_doc_length = len(tokenized_corpus[target_doc])

    print(">>> Using the functions independently")
    print(f"TF: {term_frequency(target_term, target_doc, target_doc_length, inverted_index):.4f}")
    print(f"IDF: {inverse_document_frequency(target_term, term_index, corpus_size):.4f}")
    tf_idf_val = compute_tfidf(target_term, target_doc, target_doc_length, term_index, inverted_index, corpus_size)
    print(f"TF-IDF: {tf_idf_val:.4f}")
    print()

    print(">>> Tfidf V1")
    vectorizer_1 = TfidfV1()
    matrix = vectorizer_1.fit_transform(corpus)
    print(vectorizer_1._vocabulary)
    print(matrix)
    print()

    print(">>> Tfidf V2 - calling fit and then transform")
    vectorizer_2 = TfidfV2()
    vectorizer_2.fit(corpus)
    matrix = vectorizer_2.transform(corpus)
    cols = vectorizer_2._vocabulary
    print(cols)
    print(matrix)
    print()

    print(">>> Tfidf V2 -  calling fit_transform")
    vectorizer_2 = TfidfV2()
    matrix = vectorizer_2.fit_transform(corpus)
    # sort the columns for comparison
    col_idx = [vectorizer_2._vocabulary.index(c) for c in cols]
    print(matrix[:, col_idx])
    print()

    # sanity-check
    print(">>> sklearn TfidfVectorizer -  sanity check")

    vectorizer_1 = TfidfVectorizer()
    X = vectorizer_1.fit_transform(corpus)
    print(X.toarray())
