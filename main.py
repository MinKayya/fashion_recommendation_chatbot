import requests
import os
from langchain.agents import Tool, AgentExecutor, ZeroShotAgent
from langchain.tools import tool
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import openai
from langchain.memory import ConversationBufferMemory
from config.settings import Settings
from tools.googleTrendsTool import GoogleTrendsTool
from tools.vectorSearchTool import *
from tools.weatherTool import WeatherTool
import streamlit as st

st.set_page_config(page_title="Fashion Recommendation Chatbot", layout="wide")
st.title("👗 패션 추천 챗봇")

weather = WeatherTool(Settings.WEATHER_API_KEY, city='Seoul')
google_trends = GoogleTrendsTool()

# LangChain Tool 정의
@tool
def vectordb_tool(user_request):
    """Search for fashion recommendations based on user requests and return set-based recommendations."""
    return search(user_request)

@tool
def weather_tool(city: str) -> str:
    """Get the current weather data based on the specified city."""
    return weather.get_weather()

@tool
def google_trends_tool(user_request: str) -> str:
    """Retrieve the latest trends from Google based on the given keyword."""
    return google_trends.get_trends(keywords=user_request)

tools = [
    Tool(name="RAG VectorDB Query", func=vectordb_tool, description="Queries the Fashion database."),
    Tool(name="Weather Data", func=weather_tool, description="Fetches data from Weather API."),
    Tool(name="Trends Data", func=google_trends_tool, description="Fetches data from Google Trends API.")
]

prompt = ChatPromptTemplate.from_messages([
    (
        """
        "system" : You are a fashion expert. Provide recommendations based on trends and always consider the weather according to the request.
        Adhere to the following conditions.
            Always respond in Korean.
            Always take the weather into account.
            Always Use weather_tool, google_trends_tool.
            Always use polite and gentle language.
            Recommend 2 to 3 combinations.
            Utilize all available tools to respond to the request.
            Include combinations such as tops, bottoms, and other items in your response.
        """
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

memory = ConversationBufferMemory(memory_key="chat_history")

# LangChain Agent 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.6, max_tokens=1500,)
agent = ZeroShotAgent.from_llm_and_tools(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True
)

# Agent 실행
# def run_agent():
#     print("LangChain Agent 시작! 'exit'를 입력하면 종료됩니다.")
#     while True:
#         user_request = input("요청을 입력하세요: ")
#         if user_request.lower() in ["exit", "quit", '종료']:
#             print("Agent 종료.")
#             break

#         # response = agent.run(user_request)
#         response = agent_executor.invoke({'input' : f'{user_request}'})
#         print("\n응답:\n")
#         print(response)

# if __name__ == "__main__":
#     run_agent()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

with st.container():
    user_request = st.text_input("패션 추천 요청을 입력하세요:", key="user_request")
    if st.button("추천받기"):
        if user_request.strip():
            # 사용자 요청 처리
            response = agent_executor.invoke({'input' : f'{user_request}'})

            # 'output' 키의 내용만 출력하도록 처리
            if isinstance(response, dict) and "output" in response:
                response_content = response["output"]
            else:
                response_content = response

            # 대화 기록 저장
            st.session_state["messages"].append({"role": "user", "content": user_request})
            st.session_state["messages"].append({"role": "assistant", "content": response_content})

        else:
            st.warning("요청을 입력하세요!")

# 대화 기록 표시
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(f"**👤 사용자:** {msg['content']}")
    else:
        st.markdown(f"**🤖 추천봇:** {msg['content']}")