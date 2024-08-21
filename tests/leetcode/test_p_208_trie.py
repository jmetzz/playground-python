import pytest

from data_structures.trees import TrieSymbolTableDict


def test_trie_insert_and_get(dict_trie):
    assert dict_trie.get("hello") is not None
    assert dict_trie.get("hell") is not None
    assert dict_trie.get("he") is not None
    assert dict_trie.get("heat") is not None
    assert dict_trie.get("app") is not None
    assert dict_trie.get("apple") is not None
    assert dict_trie.get("appel") is None
    assert dict_trie.get("unknown") is None


def test_trie_contains(dict_trie):
    assert dict_trie.contains("hello")
    assert not dict_trie.contains("hel")


def test_trie_size(dict_trie):
    assert dict_trie.size() == 6


def test_trie_keys_with_prefix(dict_trie):
    assert set(dict_trie.keys_with_prefix("he")) == {"hello", "hell", "he", "heat"}
    assert set(dict_trie.keys_with_prefix("hel")) == {"hello", "hell"}


def test_trie_is_valid_prefix(dict_trie):
    assert dict_trie.is_valid_prefix("he")
    assert not dict_trie.is_valid_prefix("Ha")


def test_trie_delete(dict_trie):
    dict_trie.delete("hello")
    assert not dict_trie.contains("hello")
    assert dict_trie.size() == 5
    # Test deleting a non-existent key does not alter size
    before_size = dict_trie.size()
    dict_trie.delete("nonexistent")
    assert dict_trie.size() == before_size


def test_trie_empty(dict_trie):
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
