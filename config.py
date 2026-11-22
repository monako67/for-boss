"""Настройки финального проекта из переменных окружения."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load_env_file(path: Path) -> None:
    """Минимальный загрузчик .env; не перезаписывает системные переменные."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


_load_env_file(ROOT / ".env")

GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS", "").strip()
GIGACHAT_MODEL = os.getenv("GIGACHAT_MODEL", "GigaChat-2").strip() or "GigaChat-2"
GIGACHAT_SCOPE = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS").strip()
VECTOR_STORE_PATH = ROOT / "vector_store.json"
LOG_PATH = ROOT / "agent.log"
MAX_STEPS = 2
RAG_THRESHOLD = 0.18

DOCS_DIR = ROOT / 'docs'
VECTOR_FILE = ROOT / 'vectors.json'
HISTORY_FILE = ROOT / 'chat_history.json'
LOG_FILE = ROOT / 'career_bot.log'
EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
TOP_K = 3
MIN_SIMILARITY = 0.35
EMBEDDING_BATCH_SIZE = 16
MAX_HISTORY_MESSAGES = 20


def require_credentials() -> str:
    """Вернуть учётные данные GigaChat или понятную ошибку."""
    if not GIGACHAT_CREDENTIALS:
        raise RuntimeError(
            "Не задан GIGACHAT_CREDENTIALS. Скопируйте .env.example в .env "
            "и укажите токен API."
        )
    return GIGACHAT_CREDENTIALS

