"""Внутренний ответ LLM и ограниченное состояние незавершённого запроса.

Фильтры и условия импортируются из backend, отдельного языка поиска здесь нет.
Состояние модели не является proposed_state: его собирает модуль диалога.
"""

from typing import Annotated, Literal

from app.contracts.chat import Option
from app.contracts.filter import CategoryCode, Condition, FilterV1
from app.contracts.read_api import Contract
from pydantic import Field, StrictBool, StrictStr, TypeAdapter, field_validator

Text = Annotated[StrictStr, Field(min_length=1)]


class ModelOption(Option):
    """Та же Option backend, с ограничениями, уже проверяемыми MLChat."""

    value: Annotated[StrictStr, Field(min_length=1)]
    label: Annotated[StrictStr, Field(min_length=1)]


class PartialRequest(Contract):
    operation: Literal["search_objects", "count_by_region"] | None
    category: CategoryCode | None
    conditions: list[Condition]
    missing: Annotated[list[Text], Field(min_length=1)]

    @field_validator("missing")
    @classmethod
    def meaningful_missing(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("Не указан недостающий параметр")
        return value


class ModelCandidate(Contract):
    kind: Literal["candidate"]
    new_request: StrictBool
    filter: FilterV1


class ModelClarification(Contract):
    kind: Literal["clarification"]
    new_request: StrictBool
    message: Text
    options: list[ModelOption]
    pending: PartialRequest


class ModelUnsupported(Contract):
    kind: Literal["unsupported"]
    new_request: StrictBool
    message: Text


ModelDecision = Annotated[
    ModelCandidate | ModelClarification | ModelUnsupported, Field(discriminator="kind")
]
model_decision_adapter: TypeAdapter[ModelDecision] = TypeAdapter(ModelDecision)


class PendingRequest(Contract):
    request_text: Text
    parsed: PartialRequest
    question: Text
    options: list[Option]


class State(Contract):
    version: Literal[1]
    pending_request: PendingRequest
