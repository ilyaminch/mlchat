"""Консервативные отказы до LLM; исполняемых фильтров этот модуль не строит."""

import re
from decimal import Decimal

from app.catalog.capabilities import QueryCapabilities

QUOTED = re.compile(r'"[^"\n]*"|«[^»\n]*»|“[^”\n]*”|\'[^\'\n]*\'')
CONTINUATION = re.compile(
    r"^(?:а\s+теперь\s+)?(?:оставь(?:те)?|увеличь(?:те)?|уменьши(?:те)?|"
    r"измени(?:те)?|замени(?:те)?|добавь(?:те)?|убери(?:те)?)\b|^а\s+теперь\s+только\b"
)
RATING_THRESHOLD = re.compile(
    r"\bс\s+рейтинг\w*\s+(?:(?:не\s+)?(?:выше|ниже|больше|меньше|менее|более|"
    r"равно|ровно|от)\s+)?(?P<value>[+-]?\d+(?:[.,]\d+)?)\b"
)


def unsupported_reason(
    text: str, capabilities: QueryCapabilities, *, has_pending: bool
) -> str | None:
    # Кавычки защищают значения названий. Нормализация не попадает в AST.
    normalized = QUOTED.sub(" ", text).strip().casefold().replace("ё", "е")
    if not has_pending and CONTINUATION.search(normalized):
        return "Изменение завершённого поиска пока не поддерживается. Укажите полный новый запрос."
    # Только явная конструкция «с рейтингом <сравнение> <число>». Не пытаемся
    # интерпретировать весь русский запрос или исправлять придуманные моделью числа.
    if not re.search(r"\bназвани\w*\b", normalized):
        rule = next((item for item in capabilities.fields if item.field == "rating"), None)
        if rule is not None:
            for match in RATING_THRESHOLD.finditer(normalized):
                value = Decimal(match["value"].replace(",", "."))
                below = rule.minimum is not None and value < Decimal(str(rule.minimum))
                above = rule.maximum is not None and value > Decimal(str(rule.maximum))
                if below or above:
                    return (
                        "Указанный порог рейтинга не поддерживается текущими ограничениями поиска."
                    )
    return None
