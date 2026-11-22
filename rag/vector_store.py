"""Построение, атомарное сохранение и семантический поиск."""
import json
import math
from pathlib import Path
import config
from .corpus import prepare_corpus
from .embedder import embed, embed_batch


def build_vectors(chunks):
    vectors = embed_batch([item['text'] for item in chunks])
    if len(vectors) != len(chunks):
        raise ValueError('Количество векторов не совпало с количеством чанков')
    return [dict(item, vector=vector) for item, vector in zip(chunks, vectors)]


def save_vectors(vectors, path=config.VECTOR_FILE):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(vectors, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def load_vectors(path=config.VECTOR_FILE):
    records = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(records, list):
        raise ValueError('Векторная база должна быть списком')
    for item in records:
        if not isinstance(item, dict) or not all(key in item for key in ('source','text','vector')):
            raise ValueError('Некорректная запись векторной базы')
        if not item['vector'] or not all(isinstance(value, (int,float)) and math.isfinite(value)
                                         for value in item['vector']):
            raise ValueError('Некорректный вектор')
    if len({len(item['vector']) for item in records}) > 1:
        raise ValueError('Разная размерность векторов')
    return records


def build_and_save():
    chunks = prepare_corpus()
    if not chunks:
        raise ValueError('Папка docs не содержит непустых документов')
    vectors = build_vectors(chunks)
    save_vectors(vectors)
    return vectors


def get_or_build_vectors():
    if config.VECTOR_FILE.exists():
        try:
            vectors = load_vectors()
            if vectors:
                return vectors
        except (ValueError, OSError):
            pass
    return build_and_save()


def cosine_similarity(left, right):
    if len(left) != len(right):
        raise ValueError('Размерность вопроса и базы различается, пересоберите vectors.json')
    denominator = math.sqrt(sum(value * value for value in left) * sum(value * value for value in right))
    return sum(a * b for a, b in zip(left, right)) / denominator if denominator else 0.0


def search(query, vectors, top_k=config.TOP_K, min_score=config.MIN_SIMILARITY):
    if not query.strip() or top_k < 1:
        raise ValueError('Нужны непустой вопрос и положительный top_k')
    if not vectors:
        return []
    query_vector = embed(query)
    results = [dict(item, score=cosine_similarity(query_vector, item['vector'])) for item in vectors]
    results.sort(key=lambda item: item['score'], reverse=True)
    return [item for item in results[:top_k] if item['score'] >= min_score]
