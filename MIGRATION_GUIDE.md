# Migration Guide: Provider-Agnostic LLM Support

This guide explains how to migrate your django-ai-assistant code to take advantage of the new provider-agnostic LLM support.

## What Changed?

Previously, django-ai-assistant was tightly coupled to OpenAI. Now it supports multiple LLM providers (OpenAI, Anthropic, Groq, Google, Cohere, Ollama) through a flexible provider registry system.

## Backward Compatibility

**Good news:** Your existing code will continue to work without any changes! OpenAI is still the default provider, so:

```python
class MyAssistant(AIAssistant):
    id = "my_assistant"
    name = "My Assistant"
    model = "gpt-4o-mini"
    instructions = "You are a helpful assistant."
```

This will continue to use OpenAI as before.

## Using Different Providers

### Step 1: Install the Provider Package

Install the LangChain integration for your desired provider:

```bash
# For Anthropic Claude
pip install django-ai-assistant[anthropic]

# For Groq (fast inference)
pip install django-ai-assistant[groq]

# For Google Gemini
pip install django-ai-assistant[google]

# For Cohere
pip install django-ai-assistant[cohere]

# Or install all providers at once
pip install django-ai-assistant[all-providers]
```

### Step 2: Set the API Key

Add the appropriate API key to your environment variables or Django settings:

```bash
# .env file
ANTHROPIC_API_KEY=your_anthropic_key_here
GROQ_API_KEY=your_groq_key_here
GOOGLE_API_KEY=your_google_key_here
```

### Step 3: Specify the Provider

Add the `provider` attribute to your assistant class:

```python
class MyAnthropicAssistant(AIAssistant):
    id = "my_anthropic_assistant"
    name = "My Anthropic Assistant"
    provider = "anthropic"  # Add this line
    model = "claude-3-5-sonnet-20241022"  # Use Anthropic model name
    instructions = "You are a helpful assistant."
```

## Supported Providers

| Provider | Package | Example Model | API Key Env Var |
|----------|---------|---------------|-----------------|
| OpenAI (default) | `langchain-openai` | `gpt-4o-mini` | `OPENAI_API_KEY` |
| Anthropic | `langchain-anthropic` | `claude-3-5-sonnet-20241022` | `ANTHROPIC_API_KEY` |
| Groq | `langchain-groq` | `llama-3.1-70b-versatile` | `GROQ_API_KEY` |
| Google | `langchain-google-genai` | `gemini-1.5-flash` | `GOOGLE_API_KEY` |
| Cohere | `langchain-cohere` | `command-r-plus` | `COHERE_API_KEY` |
| Ollama | `langchain-community` | `llama3` | N/A (local) |

## Advanced: Custom Providers

You can register custom providers using the provider registry:

```python
from langchain_custom_provider import ChatCustomProvider
from django_ai_assistant.providers import register_provider

# Register the custom provider
register_provider(
    "custom",  # Provider name
    ChatCustomProvider,  # LangChain chat model class
    supports_json_schema=False,  # Whether it supports OpenAI-style JSON schema
)

# Use it in your assistant
class MyCustomAssistant(AIAssistant):
    id = "my_custom_assistant"
    name = "My Custom Assistant"
    provider = "custom"
    model = "custom-model-name"
    instructions = "You are a helpful assistant."
```

## Advanced: Overriding get_llm()

If you need full control over LLM initialization, you can still override `get_llm()`:

```python
from langchain_anthropic import ChatAnthropic

class MyCustomAssistant(AIAssistant):
    id = "my_custom_assistant"
    name = "My Custom Assistant"
    model = "claude-3-5-sonnet-20241022"
    instructions = "You are a helpful assistant."

    def get_llm(self):
        return ChatAnthropic(
            model=self.model,
            temperature=0.5,
            max_tokens=1024,
            # Any custom parameters...
        )
```

## Provider-Specific Features

### Structured Outputs

The system automatically handles provider-specific structured output methods:

- **OpenAI**: Uses `json_schema` method with strict mode
- **Other providers**: Uses `json_mode` method

This is handled automatically based on the provider configuration.

### Temperature Support

Some models don't support temperature. You can set `temperature = None` to omit it:

```python
class MyAssistant(AIAssistant):
    id = "my_assistant"
    name = "My Assistant"
    provider = "custom"
    model = "custom-model"
    temperature = None  # Don't pass temperature parameter
    instructions = "You are a helpful assistant."
```

## Examples

See `example/weather/ai_assistants_multi_provider.py` for complete working examples using different providers.

## Migration Checklist

- [ ] Your existing code works without changes (OpenAI is the default)
- [ ] To use other providers:
  - [ ] Install the provider package: `pip install django-ai-assistant[provider-name]`
  - [ ] Set the API key in your environment
  - [ ] Add `provider = "provider-name"` to your assistant class
  - [ ] Update `model` to use the provider's model name
- [ ] Test your assistants with the new provider
- [ ] Update documentation for your team

## Getting Help

- Check available providers: `python -c "from django_ai_assistant.providers import list_providers; print(list_providers())"`
- See the provider documentation: https://vintasoftware.github.io/django-ai-assistant/
- Report issues: https://github.com/vintasoftware/django-ai-assistant/issues
