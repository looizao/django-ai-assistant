"""
Example AI Assistants using different LLM providers.

This module demonstrates how to use the same assistant logic with different
LLM providers (OpenAI, Anthropic, Groq, etc.).

To use these examples, install the required provider packages:
- pip install django-ai-assistant[anthropic]  # For Anthropic Claude
- pip install django-ai-assistant[groq]       # For Groq
- pip install django-ai-assistant[google]     # For Google Gemini
- pip install django-ai-assistant[all-providers]  # For all providers

And set the appropriate API keys in your environment:
- OPENAI_API_KEY (default provider)
- ANTHROPIC_API_KEY (for Anthropic)
- GROQ_API_KEY (for Groq)
- GOOGLE_API_KEY (for Google)
"""

from datetime import date

from django.conf import settings
from django.utils import timezone

import requests

from django_ai_assistant import AIAssistant, BaseModel, Field, method_tool


BASE_URL = "https://api.weatherapi.com/v1/"
TIMEOUT = 10


class WeatherAssistantBase(AIAssistant):
    """Base weather assistant with shared logic."""

    id = "weather_assistant_base"  # noqa: A003
    name = "Weather Assistant"

    def get_instructions(self):
        current_date_str = timezone.now().date().isoformat()
        return f"You are a weather bot. Use the provided functions to answer questions. Today is: {current_date_str}."

    @method_tool
    def fetch_current_weather(self, location: str) -> dict:
        """Fetch the current weather data for a location"""
        response = requests.get(
            f"{BASE_URL}current.json",
            params={
                "key": settings.WEATHER_API_KEY,
                "q": location,
            },
            timeout=TIMEOUT,
        )
        return response.json()

    class FetchForecastWeatherInput(BaseModel):
        location: str = Field(description="Location to fetch the forecast weather for")
        forecast_date: date = Field(description="Date in the format 'YYYY-MM-DD'")

    @method_tool(args_schema=FetchForecastWeatherInput)
    def fetch_forecast_weather(self, location, forecast_date) -> dict:
        """Fetch the forecast weather data for a location"""
        response = requests.get(
            f"{BASE_URL}forecast.json",
            params={
                "key": settings.WEATHER_API_KEY,
                "q": location,
                "days": 14,
                "dt": forecast_date.isoformat(),
            },
            timeout=TIMEOUT,
        )
        return response.json()


# Example 1: OpenAI GPT-4o-mini (default provider)
class WeatherAssistantOpenAI(WeatherAssistantBase):
    id = "weather_assistant_openai"  # noqa: A003
    name = "Weather Assistant (OpenAI)"
    provider = "openai"
    model = "gpt-4o-mini"


# Example 2: Anthropic Claude
class WeatherAssistantAnthropic(WeatherAssistantBase):
    id = "weather_assistant_anthropic"  # noqa: A003
    name = "Weather Assistant (Anthropic)"
    provider = "anthropic"
    model = "claude-3-5-sonnet-20241022"
    # Anthropic models may work better with lower temperatures
    temperature = 0.7


# Example 3: Groq (fast inference)
class WeatherAssistantGroq(WeatherAssistantBase):
    id = "weather_assistant_groq"  # noqa: A003
    name = "Weather Assistant (Groq)"
    provider = "groq"
    model = "llama-3.1-70b-versatile"


# Example 4: Google Gemini
class WeatherAssistantGoogle(WeatherAssistantBase):
    id = "weather_assistant_google"  # noqa: A003
    name = "Weather Assistant (Google)"
    provider = "google"
    model = "gemini-1.5-flash"


# Example 5: Custom provider registration
# If you want to use a provider not auto-registered, you can register it manually:
#
# from langchain_custom_provider import ChatCustomProvider
# from django_ai_assistant.providers import register_provider
#
# register_provider("custom", ChatCustomProvider, supports_json_schema=False)
#
# class WeatherAssistantCustom(WeatherAssistantBase):
#     id = "weather_assistant_custom"
#     name = "Weather Assistant (Custom)"
#     provider = "custom"
#     model = "custom-model-name"
