"""208. Implement Trie (Prefix Tree)
Medium

A trie (pronounced as "try") or prefix tree is a tree data structure used to
efficiently store and retrieve keys in a dataset of strings.
There are various applications of this data structure, such as autocomplete
and spellchecker.

Implement the Trie class:

Trie() Initializes the trie object.
void insert(String word) Inserts the string word into the trie.
boolean search(String word) Returns true if the string word is in the trie
(i.e., was inserted before), and false otherwise.
boolean startsWith(String prefix) Returns true if there is a previously
inserted string word that has the prefix prefix, and false otherwise.


Example 1:
Input
["Trie", "insert", "search", "search", "startsWith", "insert", "search"]
[[], ["apple"], ["apple"], ["app"], ["app"], ["app"], ["app"]]
Output
[null, null, true, false, true, null, true]

Explanation
Trie trie = new Trie();
trie.insert("apple");
trie.search("apple");   // return True
trie.search("app");     // return False
trie.startsWith("app"); // return True
trie.insert("app");
trie.search("app");     // return True


Constraints:

1 <= word.length, prefix.length <= 2000
word and prefix consist only of lowercase English letters.
At most 3 * 104 calls in total will be made to insert, search, and startsWith.
"""

from typing import Dict


class Trie:
    EOW_TOKEN = "#"

    def __init__(self):
        self._root = {}

    def insert(self, key: str) -> None:
        node = self._root
        for letter in key:
            if letter not in node:
                node[letter] = {}
            node = node[letter]
        # the end-of-word token symbolises
        # the key is added to the Trie.
        # Otherwise, the element in the path
        # only represent prefixes, but not words
        # added to the data structure
        node[self.EOW_TOKEN] = None

    def search(self, key: str) -> Dict[str, Dict]:
        node = self._root
        for letter in key:
            if letter not in node:
                # consumed all letters and
                # did not hit the end-of-word token
                return None
            node = node[letter]
        # check if the end-of-work token was found
        return self.EOW_TOKEN in node

    def startsWith(self, prefix: str) -> bool:
        node = self._root
        for letter in prefix:
            if letter not in node:
                return False
            node = node[letter]
        # processed all letter,
        # thus, the prefix is valid
        return True


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)
