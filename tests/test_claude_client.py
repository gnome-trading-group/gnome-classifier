from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from classifier.client.claude import Answer, ClaudeAsker
from classifier.runtime_config import Models


def response(text="{}", stop_reason="end_turn", input_tokens=12, output_tokens=7, thought=False):
    content = [SimpleNamespace(type="thinking", thinking="")] if thought else []
    content.append(SimpleNamespace(type="text", text=text))
    return SimpleNamespace(stop_reason=stop_reason, content=content,
                           usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens))


def test_reports_the_answer_and_its_usage():
    client = MagicMock()
    client.messages.create.return_value = response('{"a": 1}')
    asker = ClaudeAsker(lambda p: client, ["anthropic"], {"anthropic": "claude-haiku-5-5"})

    assert asker.ask_with_usage("sys", "prompt", {"type": "object"}, 100, "low") == Answer('{"a": 1}', 12, 7)
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-haiku-5-5" and kwargs["output_config"]["effort"] == "low"
    assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "thinking" not in kwargs


def test_a_thinking_setting_is_sent_when_given():
    client = MagicMock()
    client.messages.create.return_value = response()
    ClaudeAsker(lambda p: client, ["anthropic"], {"anthropic": "m"}).ask("s", "p", {}, 10, "low",
                                                                         thinking={"type": "disabled"})
    assert client.messages.create.call_args.kwargs["thinking"] == {"type": "disabled"}


def test_a_truncated_answer_raises_with_its_token_use():
    client = MagicMock()
    client.messages.create.return_value = response(stop_reason="max_tokens", output_tokens=300, thought=True)
    with pytest.raises(RuntimeError, match=r"stopped: max_tokens \(300/300 output tokens, with thinking block\)"):
        ClaudeAsker(lambda p: client, ["anthropic"], {"anthropic": "m"}).ask("s", "p", {}, 300, "low")


def test_a_truncated_answer_without_thinking_says_so():
    client = MagicMock()
    client.messages.create.return_value = response(stop_reason="max_tokens", output_tokens=10)
    with pytest.raises(RuntimeError, match=r"\(10/10 output tokens, no thinking block\)"):
        ClaudeAsker(lambda p: client, ["anthropic"], {"anthropic": "m"}).ask("s", "p", {}, 10, "low")


def test_results_are_cached_under_the_first_providers_model():
    asker = ClaudeAsker(lambda p: None, ["bedrock", "anthropic"], {"anthropic": "a", "bedrock": "anthropic.b"})
    assert asker.model == "anthropic.b"


def test_one_provider_and_fallback_serve_every_task():
    models = Models(provider="bedrock", fallback_provider="anthropic")
    assert models.providers() == ["bedrock", "anthropic"]
    assert Models(fallback_provider="anthropic").providers() == ["anthropic"]
    assert models.proposition_models["bedrock"] == models.canonicalize_models["bedrock"] == "anthropic.claude-haiku-5-5"
