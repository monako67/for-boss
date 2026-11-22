"""Разрешённые инструменты учебного агента."""

from __future__ import annotations


def calculator(left: float, right: float, operation: str) -> float:
    if operation == "+":
        return left + right
    if operation == "-":
        return left - right
    if operation == "*":
        return left * right
    if operation == "/":
        if right == 0:
            raise ZeroDivisionError("Деление на ноль невозможно")
        return left / right
    raise ValueError(f"Неизвестная операция: {operation}")

