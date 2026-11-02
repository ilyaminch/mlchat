"""Проверки native FilterV1, возможностей текущего backend и продолжения запроса."""

import json
from collections.abc import Iterable

from app.catalog.capabilities import QueryCapabilities
from app.contracts.filter import Condition, FilterV1

from mlchat.models import PartialRequest


def condition_values(condition: Condition) -> list[object]:
    data = condition.model_dump(mode="json")
    if condition.op == "between":
        return [data["lower"], data["upper"]]
    if condition.op == "in":
        return data["values"]
    return [data["value"]]


def validate_capabilities(query: FilterV1, capabilities: QueryCapabilities) -> None:
    """Capabilities могут сужать возможности установленной версии FilterV1."""
    if capabilities.schema_version != query.schema_version or capabilities.combination != "AND":
        raise ValueError("Несовместимая версия возможностей поиска")
    if query.operation not in capabilities.operations:
        raise ValueError("Операция не поддерживается")
    if query.category not in {category.code for category in capabilities.categories}:
        raise ValueError("Категория не поддерживается")
    fields = {field.field: field for field in capabilities.fields}
    for condition in query.conditions:
        rule = fields.get(condition.field)
        if (
            rule is None
            or query.category not in rule.categories
            or condition.op not in rule.operations
        ):
            raise ValueError("Поле или операция недоступны")
        for value in condition_values(condition):
            # FilterV1 уже проверил типы. Здесь проверяются более узкие runtime-лимиты.
            if type(value) in {int, float}:
                if rule.minimum is not None and value < rule.minimum:
                    raise ValueError("Значение меньше разрешённого")
                if rule.maximum is not None and value > rule.maximum:
                    raise ValueError("Значение больше разрешённого")
            if rule.values is not None and value not in rule.values:
                raise ValueError("Значение не разрешено")


def validate_partial(partial: PartialRequest, capabilities: QueryCapabilities) -> None:
    """Понятные условия должны быть выполнимы хотя бы для одной доступной категории.

    Временные варианты нужны только для валидации; категория/операция не подставляются
    в состояние и не передаются на исполнение до ответа пользователя.
    """
    operations = [partial.operation] if partial.operation else capabilities.operations
    categories = (
        [partial.category] if partial.category else [item.code for item in capabilities.categories]
    )
    for operation in operations:
        for category in categories:
            try:
                query = FilterV1.model_validate(
                    {
                        "schema_version": "1",
                        "operation": operation,
                        "category": category,
                        "conditions": [c.model_dump(mode="json") for c in partial.conditions],
                    }
                )
                validate_capabilities(query, capabilities)
                return
            except ValueError:
                continue
    raise ValueError("Уточнение содержит недопустимые условия")


def _condition_keys(conditions: Iterable[Condition]) -> set[str]:
    return {
        json.dumps(c.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, allow_nan=False)
        for c in conditions
    }


def validate_continuation(previous: PartialRequest, current: PartialRequest | FilterV1) -> None:
    """При ответе на уточнение нельзя потерять уже понятные ограничения."""
    if previous.category is not None and current.category != previous.category:
        raise ValueError("При уточнении изменилась категория")
    if previous.operation is not None and current.operation != previous.operation:
        raise ValueError("При уточнении изменилась операция")
    if not _condition_keys(previous.conditions) <= _condition_keys(current.conditions):
        raise ValueError("При уточнении потеряны условия")
    if (
        isinstance(current, FilterV1)
        and previous.category is not None
        and previous.operation is not None
        and _condition_keys(previous.conditions) == _condition_keys(current.conditions)
    ):
        raise ValueError("Недостающие параметры не уточнены")
