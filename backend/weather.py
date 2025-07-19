import requests
import os

# You can set your OpenWeatherMap API key as an environment variable
API_KEY = os.getenv("OPENWEATHER_API_KEY", "your_api_key_here")

def get_weather_alert(city: str):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        return {"error": "City not found or API error"}

    data = response.json()
    weather = data["weather"][0]["description"]
    temperature = data["main"]["temp"]
    humidity = data["main"]["humidity"]

    alert = ""
    if temperature > 35:
        alert = "High temperature warning"
    elif temperature < 10:
        alert = "Cold weather warning"
    elif humidity > 85:
        alert = "High humidity warning"
    else:
        alert = "Weather conditions are normal"

    return {
        "city": city,
        "weather": weather,
        "temperature": temperature,
        "humidity": humidity,
        "alert": alert
    }
