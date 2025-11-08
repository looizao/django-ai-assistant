"""Tests for the provider registry system."""

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from django_ai_assistant.helpers.assistants import AIAssistant
from django_ai_assistant.providers import (
    ProviderConfig,
    get_provider,
    list_providers,
    register_provider,
)


def test_openai_provider_is_registered():
    """Test that OpenAI provider is auto-registered."""
    provider = get_provider("openai")
    assert isinstance(provider, ProviderConfig)
    assert provider.llm_class == ChatOpenAI
    assert provider.supports_json_schema is True


def test_list_providers():
    """Test listing all registered providers."""
    providers = list_providers()
    assert "openai" in providers
    assert isinstance(providers, list)


def test_provider_config_create_llm():
    """Test creating an LLM instance from provider config."""
    provider = get_provider("openai")
    llm = provider.create_llm(model="gpt-4o-mini", temperature=0.5)
    assert isinstance(llm, BaseChatModel)
    assert isinstance(llm, ChatOpenAI)


def test_provider_config_create_llm_no_temperature():
    """Test creating an LLM instance without temperature."""
    provider = get_provider("openai")
    llm = provider.create_llm(model="gpt-4o-mini", temperature=None)
    assert isinstance(llm, BaseChatModel)


def test_get_provider_invalid():
    """Test getting a non-existent provider raises error."""
    with pytest.raises(ValueError, match="Provider 'nonexistent' is not registered"):
        get_provider("nonexistent")


def test_assistant_with_default_provider():
    """Test assistant uses OpenAI by default."""

    class TestAssistant(AIAssistant):
        id = "test_default_provider"  # noqa: A003
        name = "Test Assistant"
        instructions = "Test"
        model = "gpt-4o-mini"

    assistant = TestAssistant()
    assert assistant.get_provider() == "openai"
    llm = assistant.get_llm()
    assert isinstance(llm, ChatOpenAI)


def test_assistant_with_explicit_provider():
    """Test assistant can specify a different provider."""

    class TestAssistant(AIAssistant):
        id = "test_explicit_provider"  # noqa: A003
        name = "Test Assistant"
        instructions = "Test"
        provider = "openai"
        model = "gpt-4o-mini"

    assistant = TestAssistant()
    assert assistant.get_provider() == "openai"
    llm = assistant.get_llm()
    assert isinstance(llm, ChatOpenAI)


def test_custom_provider_registration():
    """Test registering a custom provider."""
    # Use ChatOpenAI as a mock custom provider for testing
    register_provider("custom_test", ChatOpenAI, supports_json_schema=True)

    provider = get_provider("custom_test")
    assert provider.llm_class == ChatOpenAI
    assert provider.supports_json_schema is True


def test_assistant_get_structured_output_llm():
    """Test that structured output LLM uses correct method based on provider."""
    from pydantic import BaseModel

    class OutputSchema(BaseModel):
        answer: str

    class TestAssistant(AIAssistant):
        id = "test_structured_output"  # noqa: A003
        name = "Test Assistant"
        instructions = "Test"
        model = "gpt-4o-mini"
        provider = "openai"
        structured_output = OutputSchema

    assistant = TestAssistant()
    llm = assistant.get_structured_output_llm()
    assert llm is not None
