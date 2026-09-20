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

model = ChatGroq(
    model="qwen/qwen3.8-27b"
)

tools = [add, multiply]

model_with_tools = model.bind_tools(tools)

response = model_with_tools.invoke("""
Use BOTH tools:
1. Multiply 25 by 4.
2. Add 10 and 20.
""")

print(response.tool_calls)

tools_by_name = {
    tool.name: tool
    for tool in tools
}


for tool_call in response.tool_calls:

    selected_tool = tools_by_name[tool_call["name"]]

    result = selected_tool.invoke(
        tool_call["args"]
    )

    print(
        tool_call["name"],
        "=>",
        result
    )