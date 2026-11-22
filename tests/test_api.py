"""API проверяется с мок-клиентом, без сети и ключей."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import llm
from chatbot import get_structured_user, get_career_advice


class TestApi(unittest.TestCase):
    def test_empty_messages(self):
        with self.assertRaises(ValueError):
            llm.chat([])

    def test_text_response(self):
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=' ответ '))])
        with patch('llm.get_client') as client:
            client.return_value.chat.return_value = response
            self.assertEqual(llm.chat([{'role':'user', 'content':'q'}]), 'ответ')

    def test_empty_response(self):
        for choices in [[], [SimpleNamespace(message=SimpleNamespace(content=''))]]:
            with patch('llm.get_client') as client:
                client.return_value.chat.return_value = SimpleNamespace(choices=choices)
                with self.assertRaises(RuntimeError):
                    llm.chat([{'role':'user','content':'q'}])

    def test_structured_user(self):
        with patch('chatbot.chat', return_value='{"name":"Иван","age":18}'):
            self.assertEqual(get_structured_user('Иван 18').age, 18)

    def test_invalid_user(self):
        from pydantic import ValidationError
        with patch('chatbot.chat', return_value='{"name":"Иван","age":-1}'):
            with self.assertRaises(ValidationError):
                get_structured_user('invalid')

    def test_career_advice(self):
        response = '{"profession":"QA","reason":"Внимательность","skills":["SQL"],"next_steps":["Тест-кейс"]}'
        with patch('chatbot.chat', return_value=response) as chat:
            self.assertEqual(get_career_advice('Люблю искать ошибки').profession, 'QA')
        self.assertIn('выбору профессии в IT', chat.call_args.args[0][0]['content'])
