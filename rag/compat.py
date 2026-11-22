"""Семантический поиск из дней 20–22."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def load_store(path=None):
    path = Path(path) if path else ROOT / 'vectors.json'
    records = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(records, list) or not records:
        raise ValueError('Хранилище пусто или повреждено')
    if all('vector' in item for item in records):
        records = [dict(source=item['source'], chunk_id=item.get('id'), text=item['text'],
                        embedding=item['vector']) for item in records]
    if {len(item['embedding']) for item in records} != {384}:
        raise ValueError('Неверная размерность embeddings')
    return records


def rag_search(query, store, top_k=3, model=None):
    if not query.strip() or top_k < 1:
        raise ValueError('Нужны непустой запрос и положительный top_k')
    if not store:
        return []
    model = model or get_model()
    vector = model.encode(query)
    scores = model.similarity(vector, [item['embedding'] for item in store])[0].tolist()
    results = [dict(source=item['source'], chunk_id=item.get('chunk_id'), text=item['text'], score=float(score))
               for item, score in zip(store, scores)]
    return sorted(results, key=lambda item: item['score'], reverse=True)[:top_k]


def filter_results(results, threshold=0.35):
    return [item for item in results if item['score'] >= threshold]


def build_context(results):
    return '\n\n'.join(f"Источник: {item['source']}, чанк: {item['chunk_id']}\n{item['text']}" for item in results)
