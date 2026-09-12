from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar
from unittest.mock import Mock

import pytest
from openai import OpenAI
from reproduction.ft_agent.responses_adapter import (
    ResponsesAdapterConfig,
    _AdapterHandler,
    api_metadata,
    responses_configuration_errors,
    routed_api_environment,
)


def test_json_response_ignores_a_disconnected_client() -> None:
    handler = object.__new__(_AdapterHandler)
    handler.close_connection = False
    handler.send_response = Mock()
    handler.send_header = Mock()
    handler.end_headers = Mock(side_effect=BrokenPipeError)

    handler._send_json(HTTPStatus.OK, {"status": "ok"})  # noqa: SLF001

    assert handler.close_connection is True


class _ResponsesHandler(BaseHTTPRequestHandler):
    payloads: ClassVar[list[dict[str, Any]]] = []

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def _send_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self._send_json(
            {
                "object": "list",
                "data": [{"id": "gpt-5.6-sol", "object": "model"}],
            },
        )

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        assert isinstance(payload, dict)
        self.payloads.append(payload)
        text = '{"ok":true}' if payload.get("text") else "OK"
        self._send_json(
            {
                "id": "resp_test",
                "object": "response",
                "created_at": 1_787_788_800,
                "status": "completed",
                "model": "gpt-5.6-sol",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text, "annotations": []}],
                    },
                ],
                "usage": {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
            },
        )


@pytest.fixture
def responses_upstream() -> Iterator[tuple[str, list[dict[str, Any]]]]:
    _ResponsesHandler.payloads = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ResponsesHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield f"http://{host}:{port}/v1", _ResponsesHandler.payloads
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _responses_environment(base_url: str) -> dict[str, str]:
    return {
        "FT_API_PROTOCOL": "responses",
        "FT_RESPONSES_MODEL": "gpt-5.6-sol",
        "OPENAI_API_BASE": base_url,
        "OPENAI_API_KEY": "not-recorded",
        "CHAT_MODEL": "gpt-5.2",
        "FT_STRONG_MODELS": '["gpt-5"]',
        "FT_WEAK_MODELS": '["gpt-4o-mini"]',
    }


def test_configuration_check_discovers_served_model(
    responses_upstream: tuple[str, list[dict[str, Any]]],
) -> None:
    base_url, _ = responses_upstream
    environment = _responses_environment(base_url)

    assert responses_configuration_errors(environment) == []
    environment["FT_RESPONSES_MODEL"] = "missing-model"
    assert "is not advertised" in responses_configuration_errors(environment)[0]


def test_public_metadata_redacts_url_credentials() -> None:
    environment = _responses_environment("http://user:secret@localhost:8313/v1?token=secret#fragment")

    metadata = api_metadata(environment)

    assert metadata["upstream_base"] == "http://localhost:8313/v1"
    assert metadata["fidelity"] == "model-adapted"
    assert "secret" not in json.dumps(metadata)


def test_routed_environment_translates_sdk_requests(
    responses_upstream: tuple[str, list[dict[str, Any]]],
) -> None:
    base_url, payloads = responses_upstream

    with routed_api_environment(_responses_environment(base_url)) as (runtime, metadata):
        assert runtime["CHAT_MODEL"] == "openai/gpt-5.2"
        assert json.loads(runtime["FT_STRONG_MODELS"]) == ["openai/gpt-5"]
        assert json.loads(runtime["FT_WEAK_MODELS"]) == ["openai/gpt-4o-mini"]
        assert runtime["FT_JUDGE_API_BASE"] == runtime["OPENAI_API_BASE"]
        assert metadata["fidelity"] == "model-adapted"

        client = OpenAI(api_key=runtime["OPENAI_API_KEY"], base_url=runtime["OPENAI_API_BASE"])
        response = client.chat.completions.create(
            model="gpt-5.2",
            messages=[{"role": "user", "content": "Return JSON."}],
            response_format={"type": "json_object"},
            temperature=1,
        )
        assert response.choices[0].message.content == '{"ok":true}'

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say OK."}],
            stream=True,
        )
        assert "".join(chunk.choices[0].delta.content or "" for chunk in stream) == "OK"

    assert [payload["model"] for payload in payloads] == ["gpt-5.6-sol", "gpt-5.6-sol"]
    assert payloads[0]["text"] == {"format": {"type": "json_object"}}
    assert "temperature" not in payloads[0]


def test_native_chat_configuration_is_unchanged() -> None:
    environment = {
        "OPENAI_API_BASE": "https://user:secret@example.test/v1?token=secret",
        "CHAT_MODEL": "gpt-5.2",
    }

    assert ResponsesAdapterConfig.from_environment(environment) is None
    with routed_api_environment(environment) as (runtime, metadata):
        assert runtime == environment
        assert metadata["upstream_base"] == "https://example.test/v1"
        assert metadata["fidelity"] == "provider-native"
