from dotenv import load_dotenv
from langchain.tools import tool
from langchain_groq import ChatGroq

load_dotenv()

# 1. Create a Tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b


# 2. Create LLM

model = ChatGroq(
    model="openai/gpt-oss-120b"
)


# 3. Give Tool to LLM

model_with_tools = model.bind_tools(
    [multiply]
)


# 4. Ask the LLM

response = model_with_tools.invoke(
    "What is 25 multiplied by 4?"
)

# 5. See LLM response

print(response)

# 6. See Tool Calls

print("\nTool Calls:")
print(response.tool_calls)