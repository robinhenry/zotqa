"""LLM adapter for pluggable language model backends."""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Response from an LLM."""

    content: str
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMAdapter(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None, max_tokens: int = 2048) -> LLMResponse:
        """Generate a response from the LLM."""
        pass


class AnthropicLLM(LLMAdapter):
    """Anthropic Claude adapter."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
    ):
        import anthropic

        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def generate(self, prompt: str, system: str | None = None, max_tokens: int = 2048) -> LLMResponse:
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        response = self.client.messages.create(**kwargs)

        return LLMResponse(
            content=response.content[0].text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


class MockLLM(LLMAdapter):
    """Mock LLM for testing."""

    def __init__(self, response: str = "Mock response"):
        self._response = response

    def generate(self, prompt: str, system: str | None = None, max_tokens: int = 2048) -> LLMResponse:
        return LLMResponse(
            content=self._response,
            input_tokens=len(prompt.split()),
            output_tokens=len(self._response.split()),
        )


def get_llm_adapter(provider: str = "anthropic", **kwargs) -> LLMAdapter:
    """Factory function to get an LLM adapter."""
    if provider == "anthropic":
        return AnthropicLLM(**kwargs)
    elif provider == "mock":
        return MockLLM(**kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
