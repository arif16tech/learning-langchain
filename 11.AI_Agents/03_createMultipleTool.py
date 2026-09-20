from dotenv import load_dotenv
from langchain.tools import tool
from langchain_groq import ChatGroq

load_dotenv()


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@tool
def subtract(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b


model = ChatGroq(
    model="openai/gpt-oss-120b"
)


# Give multiple tools to LLM
tools = [add, multiply, subtract]

model_with_tools = model.bind_tools(tools)


# Ask question
response = model_with_tools.invoke(
    "Use the multiply tool to calculate 25 multiplied by 4. Do not calculate it yourself."
)


# See what tool LLM selected
print("Tool Calls:")
print(response.tool_calls)


# Create tool lookup
tools_by_name = {
    tool.name: tool
    for tool in tools
}


# Get first tool call
tool_call = response.tool_calls[0]


# Find selected tool
selected_tool = tools_by_name[tool_call["name"]]


# Execute selected tool
result = selected_tool.invoke(
    tool_call["args"]
)


print("\nTool Result:")
print(result)