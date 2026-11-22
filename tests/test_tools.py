"""Изолированные проверки инструментов дня 24."""
import unittest
from tools import calculator
from rag import rag_search, filter_results, build_context


class FakeRow:
    def tolist(self):
        return [0.2, 0.9]


class FakeModel:
    def encode(self, text):
        return [1.0]

    def similarity(self, query, embeddings):
        return [FakeRow()]


class TestCalculator(unittest.TestCase):
    def test_add(self):
        self.assertEqual(calculator(2, 3, '+'), 5)

    def test_subtract(self):
        self.assertEqual(calculator(10, 4, '-'), 6)

    def test_multiply(self):
        self.assertEqual(calculator(3, 5, '*'), 15)

    def test_divide(self):
        self.assertEqual(calculator(8, 2, '/'), 4)

    def test_negative(self):
        self.assertEqual(calculator(-2, 3, '*'), -6)

    def test_fraction(self):
        self.assertAlmostEqual(calculator(2.5, 4, '+'), 6.5)

    def test_zero_division(self):
        with self.assertRaises(ZeroDivisionError):
            calculator(10, 0, '/')

    def test_unknown_operation(self):
        with self.assertRaises(ValueError):
            calculator(1, 2, 'exec')


class TestRagTools(unittest.TestCase):
    def test_search_orders_limits_and_preserves_source(self):
        store = [dict(source='a', chunk_id=0, text='a', embedding=[1]),
                 dict(source='b', chunk_id=1, text='b', embedding=[2])]
        results = rag_search('QA', store, top_k=1, model=FakeModel())
        self.assertEqual(results[0]['source'], 'b')
        self.assertEqual(len(results), 1)

    def test_empty_store(self):
        self.assertEqual(rag_search('QA', []), [])

    def test_invalid_input(self):
        for query, top_k in [('', 3), ('q', 0)]:
            with self.assertRaises(ValueError):
                rag_search(query, [], top_k)

    def test_threshold_boundary(self):
        self.assertEqual(filter_results([{'score': .35}, {'score': .34}], .35), [{'score': .35}])

    def test_context(self):
        self.assertIn('QA.txt', build_context([dict(source='QA.txt', chunk_id=1, text='Тестирование')]))
