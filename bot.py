"""Консультант варианта 17 с RAG, агентом и сохраняемой историей."""
import json
import logging
from pathlib import Path
import config
from llm import chat
from logger_config import setup_logging
from prompts import SYSTEM_PROMPT, RAG_USER_TEMPLATE, NO_CONTEXT_MESSAGE
from rag import get_or_build_vectors, search
from chatbot import Conversation, get_career_advice, get_structured_user
from agent import choose_action, run_agent
from tools import calculator

logger = logging.getLogger(__name__)


class CareerBot:
    def __init__(self):
        self.vectors = []
        self.history = self._load_history()

    def _load_history(self):
        path = config.HISTORY_FILE
        if not path.exists():
            return []
        try:
            history = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(history, list):
                raise ValueError('История должна быть списком')
            if not all(isinstance(item, dict) and item.get('role') in {'user','assistant'}
                       and isinstance(item.get('content'), str) for item in history):
                raise ValueError('Неправильная запись истории')
            return history
        except (ValueError, OSError) as error:
            logger.warning('History load error_type=%s', type(error).__name__)
            return []

    def _save_history(self):
        path = Path(config.HISTORY_FILE)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.history, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(path)

    def _remember(self, question, answer):
        self.history.extend([{'role':'user','content':question}, {'role':'assistant','content':answer}])
        self._save_history()

    def _conversation(self):
        conversation = Conversation()
        for index in range(0, len(self.history) - 1, 2):
            question, answer = self.history[index:index + 2]
            if question['role'] == 'user' and answer['role'] == 'assistant':
                conversation.add(question['content'], answer['content'])
        return conversation

    def _ask_gigachat_with_rag(self, question):
        relevant = search(question, self.vectors)
        if not relevant:
            return NO_CONTEXT_MESSAGE
        context = '\n\n'.join(f"[{item['source']}] {item['text']}" for item in relevant)
        messages = [{'role':'system','content':SYSTEM_PROMPT}]
        messages.extend(self.history[-config.MAX_HISTORY_MESSAGES:])
        messages.append({'role':'user','content':RAG_USER_TEMPLATE.format(context=context, question=question)})
        answer = chat(messages)
        sources = ', '.join(dict.fromkeys(item['source'] for item in relevant))
        return answer + '\nИсточники: ' + sources

    def answer(self, question, mode='rag'):
        if not question.strip():
            raise ValueError('Запрос пустой')
        decision = choose_action(question, use_llm=False)
        if decision.action == 'calculator':
            answer = f'Результат: {calculator(decision.a, decision.b, decision.operation):g}'
        elif question.lower() in {'привет', 'привет!', 'поздоровайся', 'кто ты?'}:
            answer = 'Здравствуйте! Я консультант по выбору профессии в IT. Какие направления вам интересны?'
        elif mode == 'agent':
            store = [dict(source=item['source'], chunk_id=item.get('id'), text=item['text'],
                          embedding=item['vector']) for item in self.vectors]
            run = run_agent(question, store, conversation=self._conversation())
            answer = run.answer
            logger.info('steps=%s actions=%s', run.steps, run.action_history)
        else:
            answer = self._ask_gigachat_with_rag(question)
        self._remember(question, answer)
        return answer

    def run(self):
        setup_logging()
        self.vectors = get_or_build_vectors()
        print('Вариант 17 — консультант по выбору профессии в IT')
        print('Команды: история, очистить, профессия, пользователь, калькулятор, режим агент, режим rag, выход')
        mode = 'rag'
        while True:
            try:
                question = input('\nВы: ').strip()
                command = question.lower()
                if command in {'выход','exit','quit'}:
                    break
                if not question:
                    continue
                if command == 'история':
                    for item in self.history:
                        print(('Вы: ' if item['role'] == 'user' else 'Бот: ') + item['content'])
                    if not self.history:
                        print('История пуста.')
                    continue
                if command == 'очистить':
                    self.history.clear()
                    self._save_history()
                    print('История очищена.')
                    continue
                if command in {'режим агент','режим rag','режим чат'}:
                    mode = 'agent' if command == 'режим агент' else 'rag'
                    print('Режим', mode)
                    continue
                if command == 'пользователь':
                    print(get_structured_user(input('Опишите пользователя: ')).model_dump_json(indent=2))
                    continue
                if command == 'профессия':
                    description = input('Опишите интересы и опыт: ')
                    answer = get_career_advice(description).model_dump_json(indent=2)
                    self._remember(description, answer)
                    print(answer)
                    continue
                if command == 'калькулятор':
                    question = input('Выражение, например 12 * 5: ').strip()
                    if choose_action(question, use_llm=False).action != 'calculator':
                        print('Введите два числа и операцию +, -, * или /.')
                        continue
                print('Бот:', self.answer(question, mode))
            except (EOFError, KeyboardInterrupt):
                break
            except Exception as error:
                logger.error('Bot error_type=%s', type(error).__name__)
                print(f'Ошибка ({type(error).__name__}). Подробности в career_bot.log.')
