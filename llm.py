"""Единственная точка обращения к GigaChat API."""

from __future__ import annotations

from typing import Any

from config import GIGACHAT_MODEL, GIGACHAT_SCOPE, require_credentials


_client: Any | None = None


def get_client() -> Any:
    """Лениво создать клиент; секрет никогда не записывается в лог."""
    global _client
    if _client is None:
        try:
            from gigachat import GigaChat
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Библиотека gigachat не установлена. Выполните: pip install -r requirements.txt"
            ) from exc
        _client = GigaChat(
            credentials=require_credentials(),
            scope=GIGACHAT_SCOPE,
            model=GIGACHAT_MODEL,
            verify_ssl_certs=False,
            timeout=30,
        )
    return _client


def chat(messages: list[dict[str, str]]) -> str:
    """Получить текстовый ответ модели, указанной в .env."""
    if not messages:
        raise ValueError("Список сообщений не может быть пустым")
    response = get_client().chat({"model": GIGACHAT_MODEL, "messages": messages})
    if not response.choices or not response.choices[0].message.content:
        raise RuntimeError("GigaChat вернул пустой ответ")
    return response.choices[0].message.content.strip()

