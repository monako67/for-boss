"""Ленивая загрузка одной модели для документов и вопросов."""
from config import EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE

_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed(text):
    return get_model().encode(text, normalize_embeddings=True).tolist()


def embed_batch(texts, batch_size=EMBEDDING_BATCH_SIZE):
    if not texts:
        return []
    vectors = get_model().encode(texts, batch_size=batch_size,
                                 normalize_embeddings=True, show_progress_bar=False)
    return [vector.tolist() for vector in vectors]
