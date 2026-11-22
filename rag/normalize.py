"""Нормализация текста для диагностики корпуса."""
import re


def normalize_text(text):
    return re.findall(r'\w+', text.lower(), flags=re.UNICODE)
