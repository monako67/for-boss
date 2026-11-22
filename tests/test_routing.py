import unittest
from agent import choose_action
from scenarios import CASES


class TestRouting(unittest.TestCase):
    def test_eighteen_expected_actions(self):
        for query, expected in CASES:
            with self.subTest(query=query):
                self.assertEqual(choose_action(query, use_llm=False).action, expected)
