from langchain_community.tools.google_trends import GoogleTrendsQueryRun
from langchain_community.utilities.google_trends import GoogleTrendsAPIWrapper
import os

class GoogleTrendsTool:
    def __init__(self):
        """
        Google Trends Tool 초기화
        """
        self.tool = GoogleTrendsQueryRun(api_wrapper=GoogleTrendsAPIWrapper())

    def get_trends(self, keywords):
        """
        Google Trends 데이터를 반환
        """
        try:
            response = self.tool.run(keywords)
            return response
        except Exception as e:
            return f"Google Trends 데이터를 가져오는 중 오류가 발생했습니다: {e}"
