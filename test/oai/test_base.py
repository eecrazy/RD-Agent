
from rdagent.oai.backend.base import APIBackend


class MockBackend:
    def __init__(self):
        self.messages = []

    def _add_json_in_prompt(self, new_messages):
        self.messages.append("JSON_ADDED")


class StaticCompletionBackend(APIBackend):
    def __init__(self, response: str) -> None:
        super().__init__(
            use_chat_cache=False,
            dump_chat_cache=False,
            use_embedding_cache=False,
            dump_embedding_cache=False,
        )
        self.response = response
        self.calls = 0

    def supports_response_schema(self) -> bool:
        return False

    def _calculate_token_from_messages(self, messages: list[dict[str, object]]) -> int:
        return 0

    def _create_embedding_inner_function(self, input_content_list: list[str]) -> list[list[float]]:
        return []

    def _create_chat_completion_inner_function(
        self,
        messages: list[dict[str, object]],
        response_format: object = None,
        *args: object,
        **kwargs: object,
    ) -> tuple[str, str]:
        self.calls += 1
        return self.response, "stop"


def test_json_added_once():
    backend = MockBackend()
    try_n = 3
    json_added = False
    new_messages = ["msg1"]

    for _ in range(try_n):
        if not json_added:
            backend._add_json_in_prompt(new_messages)
            json_added = True

    assert backend.messages.count("JSON_ADDED") == 1


def test_code_response_with_embedded_markdown_fences_is_not_truncated() -> None:
    response = '''```python
PROMPT = """
Return exactly:
```json
{"answer": "A"}
```
Never start an inline ``` fence.
"""
print(PROMPT)
```'''
    backend = StaticCompletionBackend(response)

    parsed = backend._create_chat_completion_auto_continue(
        messages=[{"role": "user", "content": "write code"}],
        code_block_language="python",
    )

    assert parsed == '''PROMPT = """
Return exactly:
```json
{"answer": "A"}
```
Never start an inline ``` fence.
"""
print(PROMPT)'''
    assert backend.calls == 1
