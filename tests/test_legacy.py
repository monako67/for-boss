import unittest
from unittest.mock import patch
from agent import AgentDecision, run_agent
from chatbot import Conversation
from pipeline import normalize, split_text
from tools import calculator


class IntegrationTests(unittest.TestCase):
    def test_history_and_clear(self):
        history = Conversation()
        history.add('Меня зовут Иван', 'Приятно познакомиться')
        self.assertEqual(history.messages('Как меня зовут?')[1]['content'], 'Меня зовут Иван')
        history.clear()
        self.assertEqual(len(history.messages('Привет')), 2)

    def test_direct_answer_receives_history(self):
        history = Conversation()
        history.add('Я Иван', 'Привет')
        with patch('agent.chat', return_value='Иван') as chat:
            run = run_agent('Моё имя?', [], conversation=history,
                            selector=lambda *args: AgentDecision('direct_answer'))
        self.assertEqual(run.answer, 'Иван')
        self.assertEqual(chat.call_args.args[0][1]['content'], 'Я Иван')

    def test_calculator(self):
        for a, b, op, expected in [(2,3,'+',5),(8,3,'-',5),(2,3,'*',6),(8,2,'/',4),(-2,3,'*',-6),(2.5,2,'*',5)]:
            self.assertEqual(calculator(a,b,op), expected)
        with self.assertRaises(ZeroDivisionError):
            calculator(1,0,'/')

    def test_pipeline(self):
        self.assertEqual(normalize('  текст   здесь\n\n строка'), 'текст здесь\nстрока')
        self.assertEqual(split_text('a b c d e', 3, 1), ['a b c','c d e'])

    def test_no_data_retry_and_repeat_stop(self):
        with patch('agent.rag_search', return_value=[]):
            run = run_agent('вопрос', [], selector=lambda *args: AgentDecision('rag_search'))
        self.assertEqual(run.steps, 2)
        self.assertEqual(run.action_history, ['rag_search','rag_search'])
        self.assertIn('не найдена', run.answer)

    def test_unknown_action(self):
        run = run_agent('q', [], selector=lambda *args: AgentDecision('delete'))
        self.assertIn('запрещённое', run.answer)

    def test_rag_context_and_sources(self):
        result = dict(source='git.txt', chunk_id=0, text='git status', score=0.9)
        with patch('agent.rag_search', return_value=[result]), patch('agent.chat', return_value='git status'):
            run = run_agent('Git?', [], rag_only=True)
        self.assertEqual(run.sources, ['git.txt'])

    def test_empty_and_steps(self):
        with self.assertRaises(ValueError):
            run_agent('', [])
        with self.assertRaises(ValueError):
            run_agent('q', [], max_steps=3)

if __name__ == '__main__':
    unittest.main(verbosity=2)
