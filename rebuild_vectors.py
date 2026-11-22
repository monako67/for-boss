"""Пересобрать vectors.json после изменения документов docs/."""
from rag import build_and_save

if __name__ == '__main__':
    print('Создано векторов:', len(build_and_save()))
