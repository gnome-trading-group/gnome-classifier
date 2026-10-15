from unittest.mock import MagicMock

from classifier.client import BedrockConverseClient


def _response(content, stop_reason="end_turn"):
    return {
        "output": {"message": {"content": content}},
        "stopReason": stop_reason,
        "usage": {"inputTokens": 12, "outputTokens": 7},
    }


def test_converse_skips_reasoning_blocks():
    raw = MagicMock()
    raw.converse.return_value = _response([{"reasoningContent": {"reasoningText": {"text": ""}}}, {"text": "[]"}])

    result = BedrockConverseClient(raw).converse("us.amazon.nova-micro-v1:0", "prompt", 100)

    assert result.text == "[]"
    assert (result.input_tokens, result.output_tokens) == (12, 7)


def test_model_specific_request_options():
    raw = MagicMock()
    raw.converse.return_value = _response([{"text": "{}"}])
    client = BedrockConverseClient(raw)

    client.converse("global.anthropic.claude-haiku-5-5", "prompt", 100)
    assert raw.converse.call_args.kwargs["additionalModelRequestFields"] == {"thinking": {"type": "disabled"}}
    assert raw.converse.call_args.kwargs["inferenceConfig"] == {"maxTokens": 100}

    client.converse("us.amazon.nova-micro-v1:0", "prompt", 100)
    assert "additionalModelRequestFields" not in raw.converse.call_args.kwargs
    assert raw.converse.call_args.kwargs["inferenceConfig"] == {"maxTokens": 100, "temperature": 0.0}


def test_converse_many_omits_failures():
    raw = MagicMock()
    raw.converse.side_effect = lambda **kw: (
        _response([{"text": "ok"}]) if "good" in kw["messages"][0]["content"][0]["text"] else (_ for _ in ()).throw(RuntimeError("boom"))
    )

    results = BedrockConverseClient(raw).converse_many("m", {"a": "good", "b": "bad"}, 100)

    assert set(results) == {"a"}
