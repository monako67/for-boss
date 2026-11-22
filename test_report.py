"""Таблица expected_action/actual_action/PASS для локального маршрутизатора."""
import csv
from pathlib import Path
from agent import choose_action
from scenarios import CASES


def main():
    rows = []
    for query, expected in CASES:
        actual = choose_action(query, use_llm=False).action
        rows.append(dict(query=query, expected_action=expected, actual_action=actual,
                         status='PASS' if actual == expected else 'FAIL'))
    root = Path(__file__).resolve().parent
    with (root / 'agent_test_cases.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    passed = sum(row['status'] == 'PASS' for row in rows)
    report = (f'Локальная маршрутизация варианта 17\nPASS: {passed}\nFAIL: {len(rows)-passed}\n'
              f'Успешно: {passed/len(rows)*100:.1f}%\n'
              'Проверяется выбор действия. Ответы реального API оцениваются отдельно.\n')
    (root / 'test_report.txt').write_text(report, encoding='utf-8')
    print(report)
    raise SystemExit(passed != len(rows))

if __name__ == '__main__':
    main()
