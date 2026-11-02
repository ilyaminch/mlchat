"""Фабрика подключения к неизменённому backend и один async-шаг диалога."""

import asyncio
import json
import re
from time import monotonic

from app.catalog.capabilities import QueryCapabilities
from app.contracts.chat import Candidate, Clarification, DialogueState, Unsupported
from app.core.errors import ApiError
from app.llm.client import LLMClient
from pydantic import ValidationError

from mlchat.choices import resolve_category_choice
from mlchat.generation import QWEN35_9B_PROFILE, GenerationProfile, with_generation_profile
from mlchat.guards import unsupported_reason
from mlchat.models import (
    ModelCandidate,
    ModelClarification,
    PendingRequest,
    State,
    model_decision_adapter,
)
from mlchat.prompts import build_messages, response_schema
from mlchat.validation import validate_capabilities, validate_continuation, validate_partial


def _invalid_response(details: list[dict[str, str]] | None = None) -> ApiError:
    return ApiError(
        502, "LLM_INVALID_RESPONSE", "Модель вернула некорректный ответ.", details=details
    )


def _check_size(value: dict, maximum: int | float | None) -> None:
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if maximum is not None and len(encoded) > maximum:
        raise ValueError("Превышен размер состояния")


class Dialogue:
    """Без изменяемого состояния: один экземпляр обслуживает независимые диалоги."""

    _generation_profile: GenerationProfile | None = None

    async def interpret(
        self,
        *,
        text: str,
        state_snapshot: DialogueState,
        capabilities: QueryCapabilities,
        llm_client: LLMClient,
        deadline: float,
    ) -> Candidate | Clarification | Unsupported:
        if not isinstance(text, str) or not text.strip():
            raise ApiError(422, "VALIDATION_ERROR", "Введите текст сообщения.")
        maximum = capabilities.limits.get("message_characters")
        if maximum is not None and len(text) > maximum:
            raise ApiError(413, "PAYLOAD_TOO_LARGE", "Сообщение превышает допустимый размер.")
        if monotonic() >= deadline:
            raise ApiError(504, "REQUEST_TIMEOUT", "Время ожидания ответа истекло.", True)

        try:
            if not isinstance(state_snapshot, dict):
                raise ValueError("Состояние должно быть JSON-объектом")
            _check_size(state_snapshot, capabilities.limits.get("state_bytes"))
            state = State.model_validate(state_snapshot) if state_snapshot else None
            if state is not None:
                validate_partial(state.pending_request.parsed, capabilities)
        except (ValueError, TypeError, RecursionError):
            raise ApiError(
                502,
                "DIALOGUE_INVALID_RESPONSE",
                "Состояние диалога несовместимо. Начните новый диалог.",
            ) from None

        reason = unsupported_reason(text, capabilities, has_pending=state is not None)
        if reason is not None:
            return Unsupported(
                kind="unsupported",
                message=reason,
                proposed_state=state.model_dump(mode="json") if state is not None else {},
            )

        if state is not None:
            selected = resolve_category_choice(text, state.pending_request, capabilities)
            if selected is not None:
                return Candidate(
                    kind="candidate", filter=selected.model_dump(mode="json"), proposed_state={}
                )

        schema = response_schema()
        messages = build_messages(text, state, capabilities, schema)
        try:
            async with asyncio.timeout_at(deadline):
                client = (
                    with_generation_profile(llm_client, self._generation_profile)
                    if self._generation_profile is not None
                    else llm_client
                )
                raw = await client.complete(
                    messages=messages, deadline=deadline, response_schema=schema
                )
                # Даже если транспорт вернул Python-объект, запрещаем NaN/Infinity,
                # нестандартные значения и обход проверки через model_construct.
                encoded = json.dumps(raw, ensure_ascii=False, allow_nan=False)
                decision = model_decision_adapter.validate_json(encoded)
                if monotonic() >= deadline:
                    raise TimeoutError

                if isinstance(decision, ModelCandidate):
                    validate_capabilities(decision.filter, capabilities)
                    if state is not None and not decision.new_request:
                        validate_continuation(state.pending_request.parsed, decision.filter)
                    return Candidate(
                        kind="candidate",
                        filter=decision.filter.model_dump(mode="json"),
                        proposed_state={},
                    )

                if not re.search(r"[А-Яа-яЁё]", decision.message):
                    raise ValueError("Отсутствует русское сообщение")
                if isinstance(decision, ModelClarification):
                    validate_partial(decision.pending, capabilities)
                    if state is not None and not decision.new_request:
                        validate_continuation(state.pending_request.parsed, decision.pending)
                    values = [option.value for option in decision.options]
                    if len(values) != len(set(values)) or any(
                        not option.value.strip() or not option.label.strip()
                        for option in decision.options
                    ):
                        raise ValueError("Некорректные варианты уточнения")
                    if decision.pending.category is None and (
                        not values or not set(values) <= {c.code for c in capabilities.categories}
                    ):
                        raise ValueError("Для категории требуются известные варианты")
                    request_text = (
                        state.pending_request.request_text
                        if state is not None and not decision.new_request
                        else text
                    )
                    proposed = State(
                        version=1,
                        pending_request=PendingRequest(
                            request_text=request_text,
                            parsed=decision.pending,
                            question=decision.message,
                            options=decision.options,
                        ),
                    ).model_dump(mode="json")
                    _check_size(proposed, capabilities.limits.get("state_bytes"))
                    return Clarification(
                        kind="clarification",
                        message=decision.message,
                        options=decision.options,
                        proposed_state=proposed,
                    )

                proposed = (
                    state.model_dump(mode="json")
                    if state is not None and not decision.new_request
                    else {}
                )
                return Unsupported(
                    kind="unsupported", message=decision.message, proposed_state=proposed
                )
        except TimeoutError as error:
            raise ApiError(
                504, "REQUEST_TIMEOUT", "Время ожидания ответа истекло.", True
            ) from error
        except ValidationError as error:
            # Только путь и код, без содержимого пользовательского запроса/ответа.
            details = [
                {"path": ".".join(map(str, item["loc"])), "code": item["type"]}
                for item in error.errors(include_input=False, include_context=False)
            ]
            raise _invalid_response(details) from None
        except (ValueError, TypeError, RecursionError):
            # Не отражаем текст/значения модели в API и не делаем repair-вызов.
            raise _invalid_response() from None


class _Qwen35Dialogue(Dialogue):
    _generation_profile = QWEN35_9B_PROFILE


def create_dialogue() -> Dialogue:
    return Dialogue()


def create_qwen35_9b_dialogue() -> Dialogue:
    """Профиль 9B: thinking отключён явно, temperature=0 и top_p=0.8 в запросе."""
    return _Qwen35Dialogue()
