"""Параметры генерации поверх штатного клиента backend, без своего транспорта.

Представление использует родной complete: сериализацию, ошибки, deadline и лимит
ответа проверяет backend. Соединения и флаг однократного вызова принадлежат
переданному клиенту. Общий HTTP-клиент и его настройки не изменяются.
"""

from dataclasses import dataclass
from typing import cast

import httpx
from app.llm.client import LLMClient


@dataclass(frozen=True)
class GenerationProfile:
    temperature: float
    top_p: float
    reasoning_effort: str

    def request_parameters(self) -> dict[str, float | str]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "reasoning_effort": self.reasoning_effort,
        }


QWEN35_9B_PROFILE = GenerationProfile(temperature=0, top_p=0.8, reasoning_effort="none")


class _RequestParameters:
    def __init__(self, http: httpx.AsyncClient, profile: GenerationProfile):
        self._http = http
        self._profile = profile

    def stream(self, method: str, url: str, *, json: dict, **kwargs):
        payload = {**json, **self._profile.request_parameters()}
        return self._http.stream(method, url, json=payload, **kwargs)


class _ProfiledClient(LLMClient):
    def __init__(self, client: LLMClient, profile: GenerationProfile):
        # Родной __init__ сбросил бы used. Сохраняем флаг исходного клиента,
        # включая случай, когда его уже использовал другой вызывающий код.
        self._client = client
        self.settings = client.settings
        self.http = cast(httpx.AsyncClient, _RequestParameters(client.http, profile))

    @property
    def used(self) -> bool:
        return self._client.used

    @used.setter
    def used(self, value: bool) -> None:
        self._client.used = value


def with_generation_profile(client: LLMClient, profile: GenerationProfile) -> LLMClient:
    """Одно представление на шаг, с общими соединениями и native-защитой от retry."""
    return _ProfiledClient(client, profile)
