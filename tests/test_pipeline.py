import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from pipeline import normalize, split_text, build_store
from rag import load_store


class TestPipeline(unittest.TestCase):
    def test_invalid_overlap(self):
        for size, overlap in [(10,10),(10,-1),(0,0)]:
            with self.assertRaises(ValueError):
                split_text('a b', size, overlap)

    def test_empty_text(self):
        self.assertEqual(split_text(''), [])

    def test_normalization(self):
        self.assertEqual(normalize('\ufeff  QA   тест\n\n навыки '), 'QA тест\nнавыки')

    def test_store_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'store.json'
            for content in ['[]','[{"embedding":[1]}]']:
                path.write_text(content, encoding='utf-8')
                with self.assertRaises(ValueError):
                    load_store(path)

    def test_build_uses_provided_model(self):
        class Model:
            def encode(self, texts):
                return [[1.0] * 384 for _ in texts]
        with tempfile.TemporaryDirectory() as directory, patch('pipeline.ROOT', Path(directory)):
            corpus = Path(directory) / 'corpus'
            corpus.mkdir()
            (corpus / 'qa.txt').write_text('QA тестирует программы.', encoding='utf-8')
            (corpus / 'empty.txt').write_text('', encoding='utf-8')
            records = build_store(corpus, model=Model())
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]['source'], 'qa.txt')

    def test_empty_corpus(self):
        with tempfile.TemporaryDirectory() as directory, patch('pipeline.ROOT', Path(directory)):
            with self.assertRaises(ValueError):
                build_store(directory)
