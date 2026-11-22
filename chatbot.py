"""История и структурированный ответ исходного чат-бота."""
import json
from llm import chat

from prompts import SYSTEM_PROMPT


class Conversation:
    def __init__(self):
        self.turns = []

    def add(self, question, answer):
        self.turns.append({'user': question, 'bot': answer})

    def clear(self):
        self.turns.clear()

    def messages(self, question):
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
        for turn in self.turns[-20:]:
            messages.extend([{'role': 'user', 'content': turn['user']},
                             {'role': 'assistant', 'content': turn['bot']}])
        return messages + [{'role': 'user', 'content': question}]


def get_structured_user(description):
    from pydantic import BaseModel, Field

    class User(BaseModel):
        name: str
        age: int = Field(ge=0)

    response = chat([{'role': 'system', 'content': 'Верни только JSON по схеме ' +
                     json.dumps(User.model_json_schema(), ensure_ascii=False)},
                     {'role': 'user', 'content': description}])
    return User.model_validate_json(response.removeprefix('```json').removesuffix('```').strip())


def get_career_advice(description):
    from models import CareerAdvice
    response = chat([
        {'role': 'system', 'content': SYSTEM_PROMPT + ' Верни только JSON по схеме ' +
         json.dumps(CareerAdvice.model_json_schema(), ensure_ascii=False)},
        {'role': 'user', 'content': description},
    ])
    return CareerAdvice.model_validate_json(response.removeprefix('```json').removesuffix('```').strip())
