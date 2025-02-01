import requests

class WeatherTool:
    
    def __init__(self, api_key, city):
        self.api_key = api_key
        self.city = city
        self.api_url = f"http://api.openweathermap.org/data/2.5/weather?q={self.city}&appid={self.api_key}&units=metric"

    def get_weather(self):
        """
        """

        try:
            response = requests.get(self.api_url)
            response.raise_for_status()  # HTTP 에러 발생 시 예외 처리
            weather_data = response.json()

            return {
                "temperature": weather_data["main"]["temp"],       # 현재 기온
                "temp_min": weather_data["main"]["temp_min"],      # 최저 기온
                "temp_max": weather_data["main"]["temp_max"],      # 최고 기온
                "wind_speed": weather_data["wind"]["speed"],       # 풍속
                "precipitation": weather_data.get("rain", {}).get("1h", 0),  # 강수량
                "description": weather_data["weather"][0]["description"]    # 날씨 설명
            }
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"날씨 정보를 가져오는 중 오류 발생: {e}")