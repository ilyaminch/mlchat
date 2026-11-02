"""Профиль меняет параметры запроса, сохраняя гарантии native LLMClient."""

import asyncio
import json
import unittest
from time import monotonic

import httpx
from app.catalog.capabilities import capabilities
from app.core.config import Settings
from app.core.errors import ApiError
from app.llm.client import LLMClient

from mlchat.integration import create_dialogue, create_qwen35_9b_dialogue


def settings():
    return Settings(
        database_url="postgresql+asyncpg://unused@localhost/unused",
        llm_base_url="http://localhost:11434",
        llm_model="mlchat-qwen35-9b",
        llm_response_mode="json_schema",
        llm_api_key="test-only",
        _env_file=None,
    )


def response(category="school", finish_reason="stop"):
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "kind": "candidate",
                                "new_request": True,
                                "filter": {
                                    "schema_version": "1",
                                    "operation": "search_objects",
                                    "category": category,
                                    "conditions": [],
                                },
                            }
                        ),
                    },
                }
            ]
        },
    )


class GenerationProfileTests(unittest.IsolatedAsyncioTestCase):
    async def interpret(self, dialogue, client, text="Покажи школы", deadline=None):
        return await dialogue.interpret(
            text=text,
            state_snapshot={},
            capabilities=capabilities(client.settings),
            llm_client=client,
            deadline=monotonic() + 5 if deadline is None else deadline,
        )

    async def test_profile_keeps_schema_headers_timeout_and_original_one_call_guard(self):
        requests = []

        async def handler(request):
            requests.append(request)
            return response()

        config = settings()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            native = LLMClient(config, http)
            result = await self.interpret(create_qwen35_9b_dialogue(), native)
            self.assertEqual(result.filter["category"], "school")
            self.assertTrue(native.used)
            self.assertIs(native.http, http)
            self.assertFalse(http.is_closed)
            self.assertEqual(len(requests), 1)
            request = requests[0]
            body = json.loads(request.content)
            self.assertEqual(body["reasoning_effort"], "none")
            self.assertEqual(body["temperature"], 0)
            self.assertEqual(body["top_p"], 0.8)
            self.assertEqual(body["model"], "mlchat-qwen35-9b")
            self.assertEqual(body["response_format"]["type"], "json_schema")
            self.assertEqual(body["messages"][-1]["content"], "Покажи школы")
            self.assertFalse(body["stream"])
            self.assertEqual(request.headers["authorization"], "Bearer test-only")
            self.assertEqual(int(request.headers["content-length"]), len(request.content))
            self.assertGreater(request.extensions["timeout"]["read"], 0)
            self.assertLessEqual(request.extensions["timeout"]["read"], 5)
            with self.assertRaises(ApiError) as caught:
                await native.complete(messages=[], deadline=monotonic() + 5)
            self.assertEqual(caught.exception.code, "LLM_PROTOCOL_ERROR")
            self.assertEqual(len(requests), 1)

    async def test_already_used_native_client_is_not_reset_by_profile(self):
        async def handler(request):
            self.fail("Повторного запроса быть не должно")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            native = LLMClient(settings(), http)
            native.used = True
            with self.assertRaises(ApiError) as caught:
                await self.interpret(create_qwen35_9b_dialogue(), native)
            self.assertEqual(caught.exception.code, "LLM_PROTOCOL_ERROR")

    async def test_default_factory_keeps_existing_request_parameters(self):
        bodies = []

        async def handler(request):
            bodies.append(json.loads(request.content))
            return response()

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            await self.interpret(create_dialogue(), LLMClient(settings(), http))
        self.assertEqual(len(bodies), 1)
        self.assertTrue({"temperature", "top_p", "reasoning_effort"}.isdisjoint(bodies[0]))

    async def test_incomplete_native_response_is_an_error_without_retry(self):
        calls = 0

        async def handler(request):
            nonlocal calls
            calls += 1
            return response(finish_reason="length")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            native = LLMClient(settings(), http)
            with self.assertRaises(ApiError) as caught:
                await self.interpret(create_qwen35_9b_dialogue(), native)
            self.assertEqual(caught.exception.code, "LLM_INVALID_RESPONSE")
            self.assertTrue(native.used)
        self.assertEqual(calls, 1)

    async def test_concurrent_dialogues_keep_independent_native_clients(self):
        calls = []

        async def handler(request):
            text = json.loads(request.content)["messages"][-1]["content"]
            calls.append(text)
            await asyncio.sleep(0)
            return response("pharmacy" if "аптеки" in text else "school")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            dialogue = create_qwen35_9b_dialogue()
            clients = [LLMClient(settings(), http), LLMClient(settings(), http)]
            school, pharmacy = await asyncio.gather(
                self.interpret(dialogue, clients[0], "Покажи школы"),
                self.interpret(dialogue, clients[1], "Покажи аптеки"),
            )
            self.assertEqual(school.filter["category"], "school")
            self.assertEqual(pharmacy.filter["category"], "pharmacy")
            self.assertTrue(all(client.used for client in clients))
            self.assertTrue(all(client.http is http for client in clients))
        self.assertEqual(len(calls), 2)

    async def test_deadline_cancels_native_request(self):
        cancelled = asyncio.Event()

        async def handler(request):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            native = LLMClient(settings(), http)
            with self.assertRaises(ApiError) as caught:
                await self.interpret(
                    create_qwen35_9b_dialogue(), native, deadline=monotonic() + 0.05
                )
            self.assertEqual(caught.exception.code, "REQUEST_TIMEOUT")
            self.assertTrue(cancelled.is_set())
            self.assertTrue(native.used)

    async def test_external_cancellation_is_propagated(self):
        started, cancelled = asyncio.Event(), asyncio.Event()

        async def handler(request):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            native = LLMClient(settings(), http)
            task = asyncio.create_task(self.interpret(create_qwen35_9b_dialogue(), native))
            await asyncio.wait_for(started.wait(), timeout=1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(cancelled.is_set())
            self.assertTrue(native.used)
