"""Семь проверок реального семантического поиска варианта 17."""
import csv
from rag import ROOT, load_store, rag_search, filter_results

CASES = [
    ('Чем занимается тестировщик и как он описывает дефекты?', 'professions.txt'),
    ('Что создаёт frontend-разработчик?', 'professions.txt'),
    ('Какие навыки нужны начинающему тестировщику?', 'skills_and_learning.txt'),
    ('Что изучать backend-разработчику в начале обучения?', 'skills_and_learning.txt'),
    ('Как выбрать направление по своим интересам?', 'career_choice.txt'),
    ('Какая погода завтра на Луне?', None),
    ('Как приготовить картофельный суп?', None),
]


def main():
    store = load_store()
    rows = []
    for query, expected in CASES:
        accepted = filter_results(rag_search(query, store), .35)
        sources = [item['source'] for item in accepted]
        passed = expected in sources if expected else not sources
        rows.append(dict(query=query, expected_source=expected or 'нет данных',
                         actual_sources=';'.join(sources), status='PASS' if passed else 'FAIL'))
    with (ROOT / 'rag_tests.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print('PASS', sum(row['status'] == 'PASS' for row in rows), '/', len(rows))
    raise SystemExit(any(row['status'] == 'FAIL' for row in rows))

if __name__ == '__main__':
    main()
