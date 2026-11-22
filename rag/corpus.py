"""Загрузка тематических документов и чанкинг по абзацам."""
from pathlib import Path
from config import DOCS_DIR


def load_documents(docs_dir=DOCS_DIR):
    documents = []
    for path in sorted(Path(docs_dir).glob('*.txt')):
        text = path.read_text(encoding='utf-8-sig').strip()
        if text:
            documents.append({'source': path.name, 'text': text})
    return documents


def split_into_chunks(documents):
    chunks = []
    for document in documents:
        for index, line in enumerate(document['text'].splitlines()):
            if line.strip():
                chunks.append(dict(id=f"{document['source']}_{index}", source=document['source'],
                                   text=line.strip()))
    return chunks


def prepare_corpus(docs_dir=DOCS_DIR):
    return split_into_chunks(load_documents(docs_dir))
