import json
import os

import anthropic
import requests

##### Tool definition #####


def get_weather(arg: str):
    """
    Get the current weather for a given city.

    Args:
        arg (str): The city name.

    Returns:
        dict: A dictionary containing the current weather information.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if api_key is None:
        print("Error: OPENWEATHER_API_KEY environment variable is not set")
        return None
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={arg}&units=metric&appid={api_key}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting weather data: {e}")
        return None


def test_get_weather():
    """
    Test the get_weather function.
    """
    weather = get_weather("London")
    assert weather is not None
    assert "main" in weather
    assert "temp" in weather["main"]
    assert "weather" in weather
    assert len(weather["weather"]) > 0


##### Main code #####

client = anthropic.Anthropic()
response = client.beta.tools.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=1000,
    temperature=0,
    tools=[
        {
            "name": "get_weather",
            "description": "指定された場所の現在の天気を取得する",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "都市名"}},
            },
        }
    ],
    messages=[{"role": "user", "content": "サンフランシスコの天気はどうですか?"}],
)

for block in response.content:
    if block.type == "text":
        print(block.text)
    if block.type == "tool_use":
        if block.name == "get_weather":
            city = block.input["city"]  # type: ignore
            weather = get_weather(city)
            if weather is not None:
                print(f"The weather in {city} is {weather['main']['temp']}°C")
            else:
                print(f"Failed to get weather for {city}")
