"""Structured Claude requests through the configured provider (`models.provider`: the Anthropic API or Amazon Bedrock),
falling back to `models.fallback_provider` when the first is unavailable. Both clients (`anthropic.Anthropic`,
`anthropic.AnthropicBedrockMantle`) take the same Messages request, so every task that uses Claude (proposition
extraction, canonicalization, the subject merge judge) shares the providers and picks its own model per provider."""
import dataclasses
import logging
from collections.abc import Callable

import anthropic

logger = logging.getLogger(__name__)


def is_unavailable(error: Exception) -> bool:
    """Errors another provider might not have. A 400 is usually our request's fault, except running out of credit,
    which the API also reports as a 400."""
    if isinstance(error, (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.NotFoundError,
                          anthropic.InternalServerError, anthropic.APIConnectionError, anthropic.RateLimitError)):
        return True
    return isinstance(error, anthropic.BadRequestError) and "credit balance" in str(error).lower()


@dataclasses.dataclass(frozen=True)
class Answer:
    text: str
    input_tokens: int
    output_tokens: int


class ClaudeAsker:
    def __init__(self, client_for: Callable[[str], object], providers: list[str], models: dict[str, str]):
        missing = [p for p in providers if not models.get(p)]
        if not providers or missing:
            raise ValueError(f"Claude providers {providers} need a model each (missing {missing})")
        self._client_for = client_for
        self._providers = providers
        self._models = models

    @property
    def model(self) -> str:
        """The first provider's model: what results are cached under, since a fallback answers rarely."""
        return self._models[self._providers[0]]

    def ask(self, system: str, prompt: str, schema: dict, max_tokens: int, effort: str,
            thinking: dict | None = None) -> str:
        return self.ask_with_usage(system, prompt, schema, max_tokens, effort, thinking).text

    def ask_with_usage(self, system: str, prompt: str, schema: dict, max_tokens: int, effort: str,
                       thinking: dict | None = None) -> Answer:
        """The answer's JSON text, constrained to `schema`. The system prompt is cached, since every request of a
        pass repeats it. `thinking` is sent as the request's thinking setting; None leaves the model's default
        (adaptive), whose thinking tokens count against `max_tokens`."""
        last_error: Exception | None = None
        for provider in self._providers:
            try:
                return self._ask_one(provider, system, prompt, schema, max_tokens, effort, thinking)
            except Exception as error:
                if not is_unavailable(error):
                    raise
                logger.warning("Claude provider %s unavailable: %s", provider, error)
                last_error = error
        raise last_error

    def _ask_one(self, provider: str, system: str, prompt: str, schema: dict, max_tokens: int, effort: str,
                 thinking: dict | None) -> Answer:
        request = dict(
            model=self._models[provider],
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        )
        if thinking is not None:
            request["thinking"] = thinking
        response = self._client_for(provider).messages.create(**request)
        usage = getattr(response, "usage", None)
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        if response.stop_reason in ("max_tokens", "refusal"):
            # Thinking blocks come back with their text omitted, so their presence is the only sign that thinking,
            # not the answer, used up the budget.
            thought = any(b.type == "thinking" for b in response.content)
            raise RuntimeError(
                f"Claude request stopped: {response.stop_reason} ({output_tokens}/{max_tokens} output tokens, "
                f"{'with' if thought else 'no'} thinking block)"
            )
        return Answer(
            text=next(b.text for b in response.content if b.type == "text"),
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=output_tokens,
        )
