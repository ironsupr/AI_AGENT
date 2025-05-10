from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()


@tool
def calculator_tool(a:float, b:float) -> str:
    """
    Useful for performing basic arithmetic calculations.
    """
    # print(f"Tool Called: calculator_tool with arguments a={a}, b={b}")
    return f"The result of adding {a} and {b} is {a + b}."

def say_hello(name: str) -> str:
    """
    Useful for greeting a user.
    """
    return f"Hey {name}, how are you doing today?"

def main():
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

    tools = [calculator_tool, say_hello]
    agent_executor = create_react_agent(model, tools)

    print("Welcome! I'm a AI agent. Type 'quit' to exit.")
    print("You can ask me to perform calculations, answer questions, or provide information.")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break

        print("\n Assistant: ", end="")
        for chunk in agent_executor.stream(
            {
                "messages": [HumanMessage(content=user_input)]
            }
        ):
            if "agent" in chunk and "messages" in chunk["agent"]:
                for message in chunk["agent"]["messages"]:
                    print(message.content, end="")
        print()

if __name__ == "__main__":
    main()
