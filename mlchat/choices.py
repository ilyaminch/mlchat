"""Точный выбор известного кода категории не требует повторной генерации."""

from app.catalog.capabilities import QueryCapabilities
from app.contracts.filter import FilterV1

from mlchat.models import PendingRequest
from mlchat.validation import validate_capabilities

CATEGORY_MISSING = {"category", "category_code", "категория", "категория объектов", "тип объектов"}


def resolve_category_choice(
    text: str, pending: PendingRequest, capabilities: QueryCapabilities
) -> FilterV1 | None:
    partial = pending.parsed
    if (
        partial.category is not None
        or partial.operation is None
        or not all(item.strip().casefold() in CATEGORY_MISSING for item in partial.missing)
    ):
        return None
    exact = [option.value for option in pending.options if option.value == text.strip()]
    matches = exact or [
        option.value
        for option in pending.options
        if option.label.strip().casefold() == text.strip().casefold()
    ]
    if len(set(matches)) != 1:
        return None
    try:
        query = FilterV1.model_validate(
            {
                "schema_version": "1",
                "operation": partial.operation,
                "category": matches[0],
                "conditions": [item.model_dump(mode="json") for item in partial.conditions],
            }
        )
        validate_capabilities(query, capabilities)
        return query
    except ValueError:
        return None
