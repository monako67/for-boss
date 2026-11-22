"""Структурированная рекомендация по профессии для варианта 17."""
from pydantic import BaseModel, Field


class CareerAdvice(BaseModel):
    profession: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    skills: list[str] = Field(min_length=1)
    next_steps: list[str] = Field(min_length=1)
