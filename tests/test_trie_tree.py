import pytest

from data_structures.trees import TrieSymbolTableDict, TrieSymbolTableRecursive


def test_recursive_trie_insert_and_get(recursive_trie):
    assert recursive_trie.get("hello") == 1
    assert recursive_trie.get("hell") == 2
    assert recursive_trie.get("he") == 3
    assert recursive_trie.get("heat") == 4
    assert recursive_trie.get("app") == "📱"
    assert recursive_trie.get("apple") == "🍎"
    assert recursive_trie.get("appel") is None
    assert recursive_trie.get("unknown") is None


def test_recursive_trie_contains(recursive_trie):
    assert recursive_trie.contains("hello")
    assert not recursive_trie.contains("hel")


def test_recursive_trie_size(recursive_trie):
    assert recursive_trie.size() == 6


def test_recursive_trie_lazy_size(recursive_trie):
    assert recursive_trie.lazy_size() == 6


def test_recursive_trie_keys_with_prefix(recursive_trie):
    assert set(recursive_trie.keys_with_prefix("he")) == {"hello", "hell", "he", "heat"}
    assert set(recursive_trie.keys_with_prefix("hel")) == {"hello", "hell"}


def test_recursive_trie_is_valid_prefix(recursive_trie):
    assert recursive_trie.is_valid_prefix("he")
    assert not recursive_trie.is_valid_prefix("Ha")


def test_recursive_trie_delete(recursive_trie):
    recursive_trie.delete("hello")
    assert not recursive_trie.contains("hello")
    assert recursive_trie.size() == 5
    # Test deleting a non-existent key does not alter size
    before_size = recursive_trie.size()
    recursive_trie.delete("nonexistent")
    assert recursive_trie.size() == before_size


def test_recursive_trie_empty(recursive_trie):
    assert not recursive_trie.empty()
    recursive_trie.delete("hello")
    recursive_trie.delete("hell")
    recursive_trie.delete("he")
    recursive_trie.delete("heat")
    recursive_trie.delete("app")
    recursive_trie.delete("apple")
    assert recursive_trie.empty()


@pytest.mark.functional
def test_functional_trie_recursive():
    t = TrieSymbolTableRecursive()
    assert t.get("apple") is None

    t.insert("apple", "🍎")
    assert t.contains("apple") is True
    assert t.contains("app") is False

    assert t.get("apple") == "🍎"
    assert set(t.keys()) == set(["apple"])
    assert set(t.keys_with_prefix("app")) == set(["apple"])

    t.insert("app", "📱")
    assert t.get("app") == "📱"
    assert t.contains("app") is True

    assert set(t.keys_with_prefix("app")) == set(["app", "apple"])
    assert t.size() == 2
    assert t.lazy_size() == 2
    assert set(t.keys()) == set(["app", "apple"])

    assert t.is_valid_prefix("a")
    assert t.is_valid_prefix("ap")
    assert t.is_valid_prefix("app")
    assert t.is_valid_prefix("appl")
    assert t.is_valid_prefix("zeta") is False

    t.delete("app")
    assert t.size() == 1
    assert t.lazy_size() == 1
    assert set(t.keys()) == set(["apple"])

    assert t.empty() is False
    t.delete("apple")
    assert set(t.keys()) == set()
    assert t.empty()

    # try to delete non-existing key
    t.delete("app")  # not exception should be triggers, but a log.warning()


# ====
def test_dict_trie_insert_and_get(dict_trie):
    assert dict_trie.get("hello") is not None
    assert dict_trie.get("hell") is not None
    assert dict_trie.get("he") is not None
    assert dict_trie.get("heat") is not None
    assert dict_trie.get("app") is not None
    assert dict_trie.get("apple") is not None
    assert dict_trie.get("appel") is None
    assert dict_trie.get("unknown") is None


def test_dict_trie_contains(dict_trie):
    assert dict_trie.contains("hello")
    assert not dict_trie.contains("hel")


def test_dict_trie_size(dict_trie):
    assert dict_trie.size() == 6


def test_dict_trie_keys_with_prefix(dict_trie):
    assert set(dict_trie.keys_with_prefix("he")) == {"hello", "hell", "he", "heat"}
    assert set(dict_trie.keys_with_prefix("hel")) == {"hello", "hell"}


def test_dict_trie_is_valid_prefix(dict_trie):
    assert dict_trie.is_valid_prefix("he")
    assert not dict_trie.is_valid_prefix("Ha")


def test_dict_trie_delete(dict_trie):
    dict_trie.delete("hello")
    assert not dict_trie.contains("hello")
    assert dict_trie.size() == 5
    # Test deleting a non-existent key does not alter size
    before_size = dict_trie.size()
    dict_trie.delete("nonexistent")
    assert dict_trie.size() == before_size


def test_dict_trie_empty(dict_trie):
    assert not dict_trie.empty()
    dict_trie.delete("hello")
    dict_trie.delete("hell")
    dict_trie.delete("he")
    dict_trie.delete("heat")
    dict_trie.delete("app")
    dict_trie.delete("apple")
    assert dict_trie.empty()


@pytest.mark.functional
def test_functional_trie_dict():
    t = TrieSymbolTableDict()
    assert t.get("apple") is None
    t.insert("apple")
    assert t.contains("apple")
    assert t.contains("app") is False

    assert t.get("apple") is not None
    assert t.get("app") is None
    assert t.is_valid_prefix("app")
    assert list(t.keys()) == ["apple"]

    t.insert("app")
    assert set(t.keys()) == set(["app", "apple"])
    assert TrieSymbolTableDict.EOW_TOKEN in t.get("app")
    assert t.contains("app")
    assert t.size() == 2
    assert t.get("ap") is None
    assert set(t.keys_with_prefix("ap")) == set(["app", "apple"])

    t.delete("app")
    assert list(t.keys_with_prefix("ap")) == ["apple"]
    assert list(t.keys()) == ["apple"]
    assert t.size() == 1
