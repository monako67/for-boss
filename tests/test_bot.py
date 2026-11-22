import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from bot import CareerBot
from prompts import NO_CONTEXT_MESSAGE
from rag import cosine_similarity, normalize_text, split_into_chunks


class TestCareerBot(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.history_path = Path(self.temporary.name) / 'history.json'
        self.patcher = patch('config.HISTORY_FILE', self.history_path)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_history_saved_and_reloaded(self):
        bot = CareerBot()
        self.assertEqual(bot.answer('12 * 5'), 'Результат: 60')
        self.assertEqual(len(CareerBot().history), 2)

    def test_empty_context_does_not_call_api(self):
        bot = CareerBot()
        with patch('bot.search', return_value=[]), patch('bot.chat') as chat:
            self.assertEqual(bot.answer('Рецепт супа'), NO_CONTEXT_MESSAGE)
        chat.assert_not_called()

    def test_api_receives_role_context_and_history(self):
        bot = CareerBot()
        bot._remember('Люблю искать ошибки', 'Попробуйте QA')
        with patch('bot.search', return_value=[dict(source='qa.txt', text='QA проверяет ПО')]), patch('bot.chat', return_value='QA проверяет ПО') as chat:
            answer = bot.answer('Чем занимается тестировщик?')
        messages = chat.call_args.args[0]
        self.assertIn('выбору профессии в IT', messages[0]['content'])
        self.assertIn('QA проверяет ПО', messages[-1]['content'])
        self.assertEqual(messages[1]['content'], 'Люблю искать ошибки')
        self.assertIn('qa.txt', answer)

    def test_corrupt_history(self):
        self.history_path.write_text('invalid', encoding='utf-8')
        self.assertEqual(CareerBot().history, [])

    def test_cosine_non_normalized_vectors(self):
        self.assertAlmostEqual(cosine_similarity([2,0], [3,0]), 1)
        self.assertEqual(cosine_similarity([0,0], [3,0]), 0)
        with self.assertRaises(ValueError):
            cosine_similarity([1], [1,2])

    def test_normalize_and_corpus(self):
        self.assertEqual(normalize_text('QA, Тест!'), ['qa','тест'])
        self.assertEqual(len(split_into_chunks([dict(source='a',text='A\n\nB')])), 2)
