"""Проверка реальных ответов GigaChat по контексту варианта 17."""
import json
from pathlib import Path
from agent import run_agent
from chatbot import Conversation
from rag import load_store


def main():
    conversation = Conversation()
    rows = []
    store = load_store()
    for query in ['Чем занимается тестировщик?', 'Какие навыки нужны QA?']:
        run = run_agent(query, store, conversation=conversation)
        conversation.add(query, run.answer)
        rows.append(dict(query=query, action=run.action, sources=run.sources, answer=run.answer))
        print(query, run.action, run.answer)
    (Path(__file__).resolve().parent / 'live_answers.json').write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
