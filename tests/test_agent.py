"""Проверки ограничения шагов и ошибок с подменённой моделью."""
import unittest
from unittest.mock import Mock, patch
from agent import AgentDecision, choose_action, run_agent, NO_DATA_ANSWER
from chatbot import Conversation
from prompts import SYSTEM_PROMPT


class TestAgent(unittest.TestCase):
    def test_selector_error_hides_secrets(self):
        secret = 'private-api-secret'
        run = run_agent('q', [], selector=Mock(side_effect=RuntimeError(secret)))
        self.assertNotIn(secret, run.answer)
        self.assertEqual(run.steps, 1)

    def test_tool_error(self):
        run = run_agent('q', [], selector=lambda *args: AgentDecision('calculator', a=1, b=0, operation='/'))
        self.assertIn('ZeroDivisionError', run.answer)

    def test_missing_calculator_arguments(self):
        run = run_agent('q', [], selector=lambda *args: AgentDecision('calculator'))
        self.assertIn('ValueError', run.answer)

    def test_limit_exhaustion(self):
        with patch('agent.rag_search', return_value=[]):
            run = run_agent('q', [], max_steps=1, selector=lambda *args: AgentDecision('rag_search'))
        self.assertEqual(run.steps, 1)
        self.assertEqual(run.answer, NO_DATA_ANSWER)

    def test_retry_receives_observation(self):
        selector = Mock(side_effect=[AgentDecision('rag_search'), AgentDecision('direct_answer')])
        with patch('agent.rag_search', return_value=[]), patch('agent.chat') as chat:
            run = run_agent('q', [], selector=selector)
        self.assertTrue(selector.call_args_list[1].args[1])
        self.assertEqual(run.steps, 2)
        self.assertEqual(run.answer, NO_DATA_ANSWER)
        chat.assert_not_called()

    def test_repeated_failed_action_stops(self):
        selector = Mock(return_value=AgentDecision('rag_search'))
        with patch('agent.rag_search', return_value=[]) as search:
            run = run_agent('q', [], selector=selector)
        self.assertEqual(selector.call_count, 2)
        self.assertEqual(search.call_count, 1)
        self.assertEqual(run.action_history, ['rag_search', 'rag_search'])

    def test_direct_does_not_call_tools(self):
        with patch('agent.chat', return_value='Здравствуйте'), patch('agent.rag_search') as search:
            run = run_agent('Привет', [], selector=lambda *args: AgentDecision('direct_answer'))
        self.assertEqual(run.answer, 'Здравствуйте')
        search.assert_not_called()

    def test_variant_prompt_in_all_history_messages(self):
        history = Conversation()
        history.add('Люблю искать ошибки', 'Попробуйте QA')
        self.assertIn('выбору профессии в IT', history.messages('Что учить?')[0]['content'])
        self.assertIn('вариант 17', SYSTEM_PROMPT)

    def test_llm_routing_uses_variant(self):
        with patch('agent.chat', return_value='{"action":"rag_search"}') as chat:
            self.assertEqual(choose_action('Что мне подойдёт?').action, 'rag_search')
        self.assertIn('выбору профессии в IT', chat.call_args.args[0][0]['content'])

    def test_expression_routing(self):
        for expression, result in [('2,5 × 4', 10), ('8 ÷ 2', 4), ('-2 * 3', -6)]:
            decision = choose_action(expression)
            self.assertEqual(decision.action, 'calculator')
            from tools import calculator
            self.assertEqual(calculator(decision.a, decision.b, decision.operation), result)

    def test_step_and_action_logged_without_question(self):
        with self.assertLogs('agent', level='INFO') as logs:
            run_agent('my-private-query', [], selector=lambda *args: AgentDecision('calculator', a=2, b=3, operation='+'))
        self.assertIn('step=1 action=calculator', '\n'.join(logs.output))
        self.assertNotIn('my-private-query', '\n'.join(logs.output))
