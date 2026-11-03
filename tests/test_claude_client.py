from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from classifier.client.claude import Answer, ClaudeAsker
from classifier.runtime_config import Models


def response(text="{}", stop_reason="end_turn", input_tokens=12, output_tokens=7):
    return SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)],
                           usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens))


def test_reports_the_answer_and_its_usage():
    client = MagicMock()
    client.messages.create.return_value = response('{"a": 1}')
    asker = ClaudeAsker(lambda p: client, ["anthropic"], {"anthropic": "claude-haiku-5-5"})

    assert asker.ask_with_usage("sys", "prompt", {"type": "object"}, 100, "low") == Answer('{"a": 1}', 12, 7)
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-haiku-5-5" and kwargs["output_config"]["effort"] == "low"
    assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_a_truncated_answer_raises():
    client = MagicMock()
    client.messages.create.return_value = response(stop_reason="max_tokens")
    with pytest.raises(RuntimeError, match="max_tokens"):
        ClaudeAsker(lambda p: client, ["anthropic"], {"anthropic": "m"}).ask("s", "p", {}, 10, "low")


def test_results_are_cached_under_the_first_providers_model():
    asker = ClaudeAsker(lambda p: None, ["bedrock", "anthropic"], {"anthropic": "a", "bedrock": "anthropic.b"})
    assert asker.model == "anthropic.b"


def test_one_provider_and_fallback_serve_every_task():
    models = Models(provider="bedrock", fallback_provider="anthropic")
    assert models.providers() == ["bedrock", "anthropic"]
    assert Models(fallback_provider="anthropic").providers() == ["anthropic"]
    assert models.proposition_models["bedrock"] == models.canonicalize_models["bedrock"] == "anthropic.claude-haiku-5-5"
