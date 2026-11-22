"""Публичный API RAG и совместимость с агентом дня 24."""
from .corpus import load_documents, split_into_chunks, prepare_corpus
from .normalize import normalize_text
from .embedder import embed, embed_batch, get_model
from .vector_store import (build_vectors, save_vectors, load_vectors, build_and_save,
                           get_or_build_vectors, search, cosine_similarity)
from .compat import ROOT, MODEL_NAME, load_store, rag_search, filter_results, build_context
