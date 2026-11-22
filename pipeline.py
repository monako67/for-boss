"""Дни 16–20: очистка, нормализация, чанки, embeddings, хранилище."""
import argparse
import json
import re
import unicodedata
from pathlib import Path
from rag import ROOT, MODEL_NAME, get_model


def normalize(text):
    text = unicodedata.normalize('NFKC', text).replace('\ufeff', '')
    return '\n'.join(re.sub(r'\s+', ' ', line).strip() for line in text.splitlines() if line.strip())


def split_text(text, size=200, overlap=50):
    if size <= overlap or overlap < 0:
        raise ValueError('Размер чанка должен превышать неотрицательное перекрытие')
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(' '.join(words[start:start + size]))
        if start + size >= len(words):
            break
        start += size - overlap
    return chunks


def build_store(corpus=None, size=200, overlap=50, model=None):
    corpus = Path(corpus) if corpus else ROOT / 'docs'
    normalized = ROOT / 'data' / 'normalized'
    normalized.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted(corpus.glob('*.txt')):
        text = normalize(path.read_text(encoding='utf-8-sig'))
        if not text:
            continue
        (normalized / path.name).write_text(text, encoding='utf-8')
        for index, chunk in enumerate(split_text(text, size, overlap)):
            records.append(dict(source=path.name, chunk_id=index, text=chunk))
    if not records:
        raise ValueError('В корпусе нет непустых документов')
    vectors = (model or get_model()).encode([item['text'] for item in records])
    for item, vector in zip(records, vectors):
        item['embedding'] = vector.tolist() if hasattr(vector, 'tolist') else list(vector)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=Path)
    parser.add_argument('--chunk-size', type=int, default=200)
    parser.add_argument('--overlap', type=int, default=50)
    args = parser.parse_args()
    records = build_store(args.corpus, args.chunk_size, args.overlap)
    from rag import save_vectors
    save_vectors([dict(id=f"{item['source']}_{item['chunk_id']}", source=item['source'],
                       text=item['text'], vector=item['embedding']) for item in records])
    (ROOT / 'store_metadata.json').write_text(json.dumps(dict(model=MODEL_NAME,
        chunk_size=args.chunk_size, overlap=args.overlap, chunks=len(records)), indent=2), encoding='utf-8')
    print(f'Сохранено {len(records)} чанков')

if __name__ == '__main__':
    main()
