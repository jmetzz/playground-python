from mockito import mock, verify

import unittest

from hello import message


class HelloPyBuilderTest(unittest.TestCase):
    def test_should_issue_hello_message(self):
        out = mock()
        message(out)
        verify(out).write("Hello from PyBuilder project\n")
