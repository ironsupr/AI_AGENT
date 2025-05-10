from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import Tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()

def main():
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

    tools = []
    agent_executor = create_react_agent(model, tools)

    print("Welcome! I'm uoir AI agent. Type 'quit' to exit.")
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
