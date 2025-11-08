"""Provider registry for LLM providers.

This module provides a registry system to support multiple LLM providers
(OpenAI, Anthropic, Groq, etc.) in a provider-agnostic way.
"""

from typing import Any, Callable, Dict, Type

from langchain_core.language_models import BaseChatModel


class ProviderConfig:
    """Configuration for an LLM provider."""

    def __init__(
        self,
        llm_class: Type[BaseChatModel],
        supports_json_schema: bool = False,
        **default_kwargs: Any,
    ):
        """Initialize a provider configuration.

        Args:
            llm_class: The LangChain chat model class for this provider.
            supports_json_schema: Whether this provider supports json_schema method
                for structured outputs (like OpenAI's strict mode).
            **default_kwargs: Default keyword arguments to pass to the LLM constructor.
        """
        self.llm_class = llm_class
        self.supports_json_schema = supports_json_schema
        self.default_kwargs = default_kwargs

    def create_llm(
        self,
        model: str,
        temperature: float | None = None,
        **model_kwargs: Any,
    ) -> BaseChatModel:
        """Create an LLM instance with the given configuration.

        Args:
            model: The model name to use.
            temperature: The temperature to use. If None, omit the parameter.
            **model_kwargs: Additional keyword arguments to pass to the LLM constructor.

        Returns:
            An instance of the LLM.
        """
        kwargs = {**self.default_kwargs, **model_kwargs, "model": model}
        if temperature is not None:
            kwargs["temperature"] = temperature

        return self.llm_class(**kwargs)


class ProviderRegistry:
    """Registry for LLM providers."""

    def __init__(self):
        self._providers: Dict[str, ProviderConfig] = {}

    def register(
        self,
        name: str,
        llm_class: Type[BaseChatModel],
        supports_json_schema: bool = False,
        **default_kwargs: Any,
    ) -> None:
        """Register a new LLM provider.

        Args:
            name: The name of the provider (e.g., "openai", "anthropic").
            llm_class: The LangChain chat model class for this provider.
            supports_json_schema: Whether this provider supports json_schema method
                for structured outputs.
            **default_kwargs: Default keyword arguments to pass to the LLM constructor.
        """
        self._providers[name] = ProviderConfig(
            llm_class=llm_class,
            supports_json_schema=supports_json_schema,
            **default_kwargs,
        )

    def get(self, name: str) -> ProviderConfig:
        """Get a provider configuration by name.

        Args:
            name: The name of the provider.

        Returns:
            The provider configuration.

        Raises:
            ValueError: If the provider is not registered.
        """
        if name not in self._providers:
            available = ", ".join(self._providers.keys())
            raise ValueError(
                f"Provider '{name}' is not registered. "
                f"Available providers: {available}. "
                f"Make sure the provider's LangChain integration is installed "
                f"and registered before use."
            )
        return self._providers[name]

    def list_providers(self) -> list[str]:
        """List all registered provider names.

        Returns:
            A list of registered provider names.
        """
        return list(self._providers.keys())


# Global provider registry
_registry = ProviderRegistry()


def register_provider(
    name: str,
    llm_class: Type[BaseChatModel],
    supports_json_schema: bool = False,
    **default_kwargs: Any,
) -> None:
    """Register a new LLM provider in the global registry.

    Args:
        name: The name of the provider (e.g., "openai", "anthropic").
        llm_class: The LangChain chat model class for this provider.
        supports_json_schema: Whether this provider supports json_schema method
            for structured outputs.
        **default_kwargs: Default keyword arguments to pass to the LLM constructor.

    Example:
        ```python
        from langchain_anthropic import ChatAnthropic
        from django_ai_assistant.providers import register_provider

        register_provider("anthropic", ChatAnthropic, supports_json_schema=False)
        ```
    """
    _registry.register(name, llm_class, supports_json_schema, **default_kwargs)


def get_provider(name: str) -> ProviderConfig:
    """Get a provider configuration from the global registry.

    Args:
        name: The name of the provider.

    Returns:
        The provider configuration.

    Raises:
        ValueError: If the provider is not registered.
    """
    return _registry.get(name)


def list_providers() -> list[str]:
    """List all registered provider names.

    Returns:
        A list of registered provider names.
    """
    return _registry.list_providers()


# Register OpenAI provider by default
try:
    from langchain_openai import ChatOpenAI

    register_provider("openai", ChatOpenAI, supports_json_schema=True)
except ImportError:
    pass  # OpenAI not installed, skip registration


# Auto-register other common providers if available
try:
    from langchain_anthropic import ChatAnthropic

    register_provider("anthropic", ChatAnthropic, supports_json_schema=False)
except ImportError:
    pass


try:
    from langchain_groq import ChatGroq

    register_provider("groq", ChatGroq, supports_json_schema=False)
except ImportError:
    pass


try:
    from langchain_google_genai import ChatGoogleGenerativeAI

    register_provider("google", ChatGoogleGenerativeAI, supports_json_schema=False)
except ImportError:
    pass


try:
    from langchain_community.chat_models import ChatCohere

    register_provider("cohere", ChatCohere, supports_json_schema=False)
except ImportError:
    pass


try:
    from langchain_community.chat_models import ChatOllama

    register_provider("ollama", ChatOllama, supports_json_schema=False)
except ImportError:
    pass
