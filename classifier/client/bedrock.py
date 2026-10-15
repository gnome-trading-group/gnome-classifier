import dataclasses
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import boto3
from botocore.config import Config

logger = logging.getLogger(__name__)

MAX_WORKERS = 16


@dataclasses.dataclass
class ConverseResult:
    text: str
    input_tokens: int
    output_tokens: int


def bedrock_runtime_client(region_name: str | None = None) -> Any:
    # boto3's default pool of 10 connections would make some of the parallel calls reconnect.
    return boto3.client("bedrock-runtime", region_name=region_name, config=Config(max_pool_connections=MAX_WORKERS))


def _is_claude(model: str) -> bool:
    return "anthropic.claude" in model


def _additional_fields(model: str) -> dict | None:
    # Newer Claude models think by default; the output is billed but unused for a fixed-format JSON task.
    if _is_claude(model):
        return {"thinking": {"type": "disabled"}}
    return None


def _inference_config(model: str, max_tokens: int) -> dict:
    # Greedy decoding keeps categories and tags stable across re-runs. Current Claude models reject non-default
    # sampling parameters, so they keep the default.
    if _is_claude(model):
        return {"maxTokens": max_tokens}
    return {"maxTokens": max_tokens, "temperature": 0.0}


class BedrockConverseClient:
    """Bedrock Converse API wrapper so the model (Nova, Claude, ...) is a config value rather than a code path."""

    def __init__(self, client: Any, max_workers: int = MAX_WORKERS):
        self._client = client
        self._max_workers = max_workers

    def converse(self, model: str, prompt: str, max_tokens: int) -> ConverseResult:
        kwargs: dict = dict(
            modelId=model,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig=_inference_config(model, max_tokens),
        )
        extra = _additional_fields(model)
        if extra:
            kwargs["additionalModelRequestFields"] = extra
        response = self._client.converse(**kwargs)
        # Reasoning blocks can precede the answer, so pick text blocks by type rather than position.
        text = "".join(
            block["text"] for block in response["output"]["message"]["content"] if "text" in block
        )
        if response.get("stopReason") == "max_tokens":
            logger.warning("Converse response truncated at max_tokens=%d (model=%s)", max_tokens, model)
        usage = response.get("usage", {})
        return ConverseResult(
            text=text,
            input_tokens=usage.get("inputTokens", 0),
            output_tokens=usage.get("outputTokens", 0),
        )

    def converse_many(self, model: str, prompts: dict[str, str], max_tokens: int) -> dict[str, ConverseResult]:
        """Run prompts concurrently. Returns {request_id: result}; failed requests are logged and omitted."""
        results: dict[str, ConverseResult] = {}
        if not prompts:
            return results
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(self.converse, model, prompt, max_tokens): request_id
                for request_id, prompt in prompts.items()
            }
            for future in as_completed(futures):
                request_id = futures[future]
                try:
                    results[request_id] = future.result()
                except Exception as e:
                    logger.warning("Converse request %s failed: %s", request_id, e)
        return results
