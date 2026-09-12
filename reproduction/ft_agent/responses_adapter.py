"""Translate OpenAI Chat Completions requests to a Responses-only endpoint."""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit, urlunsplit

PROTOCOL_ENV = "FT_API_PROTOCOL"
UPSTREAM_BASE_ENV = "FT_RESPONSES_API_BASE"
UPSTREAM_KEY_ENV = "FT_RESPONSES_API_KEY"
SERVED_MODEL_ENV = "FT_RESPONSES_MODEL"
UPSTREAM_TIMEOUT_ENV = "FT_RESPONSES_TIMEOUT"
MODEL_ADAPTED = "model-adapted"
DEFAULT_UPSTREAM_TIMEOUT = 600.0


def _endpoint(base_url: str, resource: str) -> str:
    return f"{base_url.rstrip('/')}/{resource.lstrip('/')}"


def _public_url(value: str | None) -> str | None:
    """Remove credentials, query parameters, and fragments from recorded API URLs."""
    if not value:
        return None
    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.hostname:
        return "<invalid-url>"
    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    try:
        port = f":{parsed.port}" if parsed.port is not None else ""
    except ValueError:
        return "<invalid-url>"
    return urlunsplit((parsed.scheme, f"{host}{port}", parsed.path.rstrip("/"), "", ""))


@dataclass(frozen=True)
class ResponsesAdapterConfig:
    upstream_base: str
    served_model: str
    upstream_api_key: str | None = None
    timeout: float = DEFAULT_UPSTREAM_TIMEOUT

    @classmethod
    def from_environment(cls, environment: Mapping[str, str]) -> ResponsesAdapterConfig | None:
        protocol = environment.get(PROTOCOL_ENV, "chat_completions").strip().lower().replace("-", "_")
        if protocol in {"", "chat", "chat_completions"}:
            return None
        if protocol != "responses":
            message = f"Unsupported {PROTOCOL_ENV}={protocol!r}; expected 'chat_completions' or 'responses'"
            raise ValueError(message)

        upstream_base = environment.get(UPSTREAM_BASE_ENV) or environment.get("OPENAI_API_BASE", "")
        served_model = environment.get(SERVED_MODEL_ENV, "")
        if not upstream_base:
            message = f"{UPSTREAM_BASE_ENV} or OPENAI_API_BASE is required for Responses mode"
            raise ValueError(message)
        if not served_model:
            message = f"{SERVED_MODEL_ENV} is required for Responses mode"
            raise ValueError(message)
        parsed = urlsplit(upstream_base)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            message = f"Invalid Responses API base URL: {upstream_base!r}"
            raise ValueError(message)

        try:
            timeout = float(environment.get(UPSTREAM_TIMEOUT_ENV, DEFAULT_UPSTREAM_TIMEOUT))
        except (TypeError, ValueError) as error:
            message = f"{UPSTREAM_TIMEOUT_ENV} must be a positive number"
            raise ValueError(message) from error
        if timeout <= 0:
            message = f"{UPSTREAM_TIMEOUT_ENV} must be a positive number"
            raise ValueError(message)

        key = environment.get(UPSTREAM_KEY_ENV) or environment.get("OPENAI_API_KEY") or None
        return cls(
            upstream_base=upstream_base.rstrip("/"),
            served_model=served_model,
            upstream_api_key=key,
            timeout=timeout,
        )

    def public_metadata(self) -> dict[str, Any]:
        return {
            "protocol": "responses",
            "upstream_base": _public_url(self.upstream_base),
            "served_model": self.served_model,
            "aliases_requested_models": True,
            "fidelity": MODEL_ADAPTED,
            "litellm_model_prefix": "openai/",
        }


def api_metadata(environment: Mapping[str, str]) -> dict[str, Any]:
    config = ResponsesAdapterConfig.from_environment(environment)
    if config is None:
        return {
            "protocol": "chat_completions",
            "upstream_base": _public_url(environment.get("OPENAI_API_BASE")),
            "served_model": None,
            "aliases_requested_models": False,
            "fidelity": "provider-native",
            "litellm_model_prefix": None,
        }
    return config.public_metadata()


def responses_configuration_errors(environment: Mapping[str, str]) -> list[str]:
    try:
        config = ResponsesAdapterConfig.from_environment(environment)
    except ValueError as error:
        return [str(error)]
    if config is None:
        return []

    request = urllib.request.Request(  # noqa: S310 - config accepts only HTTP(S) URLs
        _endpoint(config.upstream_base, "models"),
    )
    if config.upstream_api_key:
        request.add_header("Authorization", f"Bearer {config.upstream_api_key}")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310 - user-configured API endpoint
            payload = json.load(response)
    except (OSError, ValueError, urllib.error.HTTPError) as error:
        return [f"Responses API model discovery failed: {error}"]

    model_ids = {item.get("id") for item in payload.get("data", []) if isinstance(item, dict)}
    if config.served_model not in model_ids:
        return [
            f"{SERVED_MODEL_ENV}={config.served_model!r} is not advertised by "
            f"{_endpoint(config.upstream_base, 'models')}",
        ]
    return []


def _response_format(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    kind = value.get("type")
    if kind == "json_object":
        return {"type": "json_object"}
    if kind != "json_schema" or not isinstance(value.get("json_schema"), dict):
        return None
    schema = value["json_schema"]
    result = {"type": "json_schema"}
    for key in ("name", "schema", "strict", "description"):
        if key in schema:
            result[key] = schema[key]
    return result


def _responses_payload(chat_payload: dict[str, Any], config: ResponsesAdapterConfig) -> dict[str, Any]:
    messages = chat_payload.get("messages")
    if not isinstance(messages, list):
        message = "'messages' must be a list"
        raise TypeError(message)
    if chat_payload.get("n", 1) not in (None, 1):
        message = "The Responses adapter supports only n=1"
        raise ValueError(message)

    payload: dict[str, Any] = {"model": config.served_model, "input": messages}
    max_output_tokens = chat_payload.get("max_completion_tokens", chat_payload.get("max_tokens"))
    if isinstance(max_output_tokens, int) and max_output_tokens > 0:
        payload["max_output_tokens"] = max_output_tokens
    if reasoning_effort := chat_payload.get("reasoning_effort"):
        payload["reasoning"] = {"effort": reasoning_effort}
    if text_format := _response_format(chat_payload.get("response_format")):
        payload["text"] = {"format": text_format}
    # gpt-5.6-sol rejects Chat Completions sampling/penalty fields, so they are
    # deliberately omitted here rather than forwarded as unsupported inputs.
    return payload


def _extract_text(payload: dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str):
        return output_text
    parts: list[str] = []
    for item in payload.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            text = content.get("text") or content.get("refusal")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _finish_reason(payload: dict[str, Any]) -> str:
    if payload.get("status") != "incomplete":
        return "stop"
    details = payload.get("incomplete_details")
    if isinstance(details, dict) and details.get("reason") == "max_output_tokens":
        return "length"
    return "stop"


def _chat_response(payload: dict[str, Any], config: ResponsesAdapterConfig) -> dict[str, Any]:
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    prompt_tokens = int(usage.get("input_tokens") or 0)
    completion_tokens = int(usage.get("output_tokens") or 0)
    return {
        "id": f"chatcmpl-adapter-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(payload.get("created_at") or time.time()),
        "model": config.served_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": _extract_text(payload)},
                "logprobs": None,
                "finish_reason": _finish_reason(payload),
            },
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": int(usage.get("total_tokens") or prompt_tokens + completion_tokens),
        },
    }


class _AdapterHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], config: ResponsesAdapterConfig) -> None:
        self.config = config
        super().__init__(address, _AdapterHandler)


class _AdapterHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _AdapterHTTPServer

    def log_message(self, _format: str, *_args: Any) -> None:
        # Prompts and authorization headers must never appear in runner logs.
        return

    def _send_json(self, status: int, payload: Any, extra_headers: Mapping[str, str] | None = None) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            if extra_headers:
                for key, value in extra_headers.items():
                    self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # Long planner calls can outlive the client timeout.  The response
            # is then no longer deliverable, but that normal disconnect should
            # not produce a second error response (and another broken pipe).
            self.close_connection = True

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            message = "Invalid Content-Length"
            raise ValueError(message) from error
        if length <= 0 or length > 64 * 1024 * 1024:
            message = "Request body is empty or too large"
            raise ValueError(message)
        payload = json.loads(self.rfile.read(length))
        if not isinstance(payload, dict):
            message = "Request body must be a JSON object"
            raise TypeError(message)
        return payload

    def _upstream_post(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        body = json.dumps(payload, ensure_ascii=False).encode()
        request = urllib.request.Request(  # noqa: S310 - config accepts only HTTP(S) URLs
            _endpoint(self.server.config.upstream_base, "responses"),
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if self.server.config.upstream_api_key:
            request.add_header("Authorization", f"Bearer {self.server.config.upstream_api_key}")
        try:
            with urllib.request.urlopen(  # noqa: S310 - user-configured API endpoint
                request,
                timeout=self.server.config.timeout,
            ) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            try:
                return error.code, json.loads(error.read())
            except (json.JSONDecodeError, UnicodeDecodeError):
                return error.code, {"error": {"message": str(error), "code": "upstream_http_error"}}

    def do_GET(self) -> None:
        path = urlsplit(self.path).path.rstrip("/")
        if path in {"", "/health"}:
            self._send_json(HTTPStatus.OK, {"status": "ok", **self.server.config.public_metadata()})
            return
        if path in {"/models", "/v1/models"}:
            self._send_json(
                HTTPStatus.OK,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.server.config.served_model,
                            "object": "model",
                            "created": 0,
                            "owned_by": "responses-adapter",
                        },
                    ],
                },
            )
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "Not found", "code": "not_found"}})

    def do_POST(self) -> None:
        path = urlsplit(self.path).path.rstrip("/")
        try:
            incoming = self._read_json()
            if path in {"/responses", "/v1/responses"}:
                incoming["model"] = self.server.config.served_model
                status, payload = self._upstream_post(incoming)
                self._send_json(status, payload, {"X-RD-Agent-Served-Model": self.server.config.served_model})
                return
            if path not in {"/v1", "/chat/completions", "/v1/chat/completions"}:
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"error": {"message": "Not found", "code": "not_found"}},
                )
                return
            responses_payload = _responses_payload(incoming, self.server.config)
            status, upstream = self._upstream_post(responses_payload)
            if status != HTTPStatus.OK or "error" in upstream:
                self._send_json(status, upstream)
                return
            chat = _chat_response(upstream, self.server.config)
            if incoming.get("stream"):
                self._send_stream(chat)
            else:
                self._send_json(
                    HTTPStatus.OK,
                    chat,
                    {"X-RD-Agent-Served-Model": self.server.config.served_model},
                )
        except (OSError, TimeoutError, TypeError, ValueError, json.JSONDecodeError) as error:
            self._send_json(
                HTTPStatus.BAD_GATEWAY,
                {"error": {"message": str(error), "code": "adapter_error"}},
            )

    def _send_stream(self, chat: dict[str, Any]) -> None:
        base = {
            "id": chat["id"],
            "object": "chat.completion.chunk",
            "created": chat["created"],
            "model": chat["model"],
        }
        chunks = [
            {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]},
            {
                **base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": chat["choices"][0]["message"]["content"]},
                        "finish_reason": None,
                    },
                ],
            },
            {
                **base,
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": chat["choices"][0]["finish_reason"]},
                ],
                "usage": chat["usage"],
            },
        ]
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-RD-Agent-Served-Model", self.server.config.served_model)
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True


class ResponsesChatAdapter:
    def __init__(self, config: ResponsesAdapterConfig, host: str = "127.0.0.1", port: int = 0) -> None:
        self.config = config
        self._server = _AdapterHTTPServer((host, port), config)
        self._thread = threading.Thread(target=self._server.serve_forever, name="responses-chat-adapter", daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    @property
    def chat_completions_url(self) -> str:
        return _endpoint(self.base_url, "chat/completions")

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def litellm_model_name(model: str, metadata: Mapping[str, Any]) -> str:
    """Force LiteLLM's OpenAI provider while preserving the requested model alias."""
    if metadata.get("protocol") != "responses" or model.startswith("openai/"):
        return model
    return f"openai/{model}"


def _route_model_pool(value: str, metadata: Mapping[str, Any]) -> str:
    try:
        models = json.loads(value)
    except json.JSONDecodeError:
        return value
    if not isinstance(models, list) or not all(isinstance(model, str) for model in models):
        return value
    return json.dumps([litellm_model_name(model, metadata) for model in models])


@contextlib.contextmanager
def routed_api_environment(
    environment: Mapping[str, str] | None = None,
) -> Iterator[tuple[dict[str, str], dict[str, Any]]]:
    runtime = dict(os.environ if environment is None else environment)
    config = ResponsesAdapterConfig.from_environment(runtime)
    if config is None:
        yield runtime, api_metadata(runtime)
        return

    adapter = ResponsesChatAdapter(config)
    adapter.start()
    runtime["OPENAI_API_BASE"] = adapter.base_url
    runtime["OPENAI_BASE_URL"] = adapter.base_url
    # OpenAI SDK clients, including OpenCompass's judge, append
    # /chat/completions to this base URL themselves.
    runtime["FT_JUDGE_API_BASE"] = adapter.base_url
    metadata = {**config.public_metadata(), "adapter_base": _public_url(adapter.base_url)}
    if model := runtime.get("CHAT_MODEL"):
        runtime["CHAT_MODEL"] = litellm_model_name(model, metadata)
    for pool_name in ("FT_STRONG_MODELS", "FT_WEAK_MODELS"):
        if pool := runtime.get(pool_name):
            runtime[pool_name] = _route_model_pool(pool, metadata)
    try:
        yield runtime, metadata
    finally:
        adapter.close()
