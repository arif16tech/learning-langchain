from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage

load_dotenv()


# 1. Create tools

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


# 2. Create LLM

model = ChatGroq(
    model="openai/gpt-oss-120b"
)


# 3. Create Agent with multiple tools

agent = create_agent(
    model=model,
    tools=[add, multiply]
)


# 4. Run Agent

user_message = HumanMessage(
    content="What is 25 multiplied by 4?"
)

response = agent.invoke({
    "messages": [
        user_message
    ]
})


# 5. Print final answer

print("\n----- Final Response -----")
print(response["messages"][-1].content)