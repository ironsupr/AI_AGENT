import streamlit as st
import time # Added for typewriter effect
import requests # Added for web access tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool # Import the 'tool' decorator
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()


@tool # Ensure 'tool' is imported and decorator is uncommented
def calculator_tool(a:float, b:float) -> str:
    """
    Useful for performing basic arithmetic calculations.
    """
    # print(f"Tool Called: calculator_tool with arguments a={a}, b={b}")
    return f"The result of adding {a} and {b} is {a + b}."

@tool # Add the 'tool' decorator
def say_hello(name: str) -> str:
    """
    Useful for greeting a user.
    """
    return f"Hey {name}, how are you doing today? Ayoo check"

@tool
def web_search_tool(url: str) -> str:
    """
    Useful for fetching content from a given URL.
    Provide the full URL including http or https.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.text
    except requests.exceptions.RequestException as e:
        return f"Error fetching URL: {e}"

@st.cache_resource
def load_agent():
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0) 
    tools = [calculator_tool, say_hello, web_search_tool] 
    agent_executor = create_react_agent(model, tools)
    return agent_executor

def main():
    st.title("AI Agent Chat")

    agent_executor = load_agent()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("What is up?")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            # Simulate typewriter effect
            for chunk in agent_executor.stream(
                {
                    "messages": [HumanMessage(content=prompt)]
                }
            ):
                if "agent" in chunk and "messages" in chunk["agent"]:
                    for message_part in chunk["agent"]["messages"]:
                        for char in message_part.content:
                            full_response += char
                            response_placeholder.markdown(full_response + "▌")
                            time.sleep(0.02)  # Adjust speed of typewriter effect
            response_placeholder.markdown(full_response) # Display final response without cursor
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()