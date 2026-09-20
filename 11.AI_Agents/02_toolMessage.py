from dotenv import load_dotenv

from langchain.tools import tool
from langchain.messages import ToolMessage, HumanMessage
from langchain_groq import ChatGroq


load_dotenv()

# 1. Create Tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b


# 2. Create LLM

model = ChatGroq(
    model="openai/gpt-oss-120b"
)

# 3. Give tool to LLM

model_with_tools = model.bind_tools([multiply])


# 4. Ask LLM

user_message = HumanMessage(
    content="Use the multiply tool to calculate 25 multiplied by 4."
)

response = model_with_tools.invoke([
    user_message
])



# 5. Check Tool Call

print("TOOL CALL:")
print(response.tool_calls)


if response.tool_calls:

    # 6. Execute Tool

    tool_call = response.tool_calls[0]

    tool_result = multiply.invoke(
        tool_call["args"]
    )

    print("\nTOOL RESULT:")
    print(tool_result)

    # 7. Create ToolMessage

    tool_message = ToolMessage(
        content=str(tool_result),
        tool_call_id=tool_call["id"]
    )


    # 8. Send result back to LLM

    final_response = model_with_tools.invoke([
        user_message,
        response,
        tool_message
    ])


    # 9. Final Answer

    print("\nFINAL ANSWER:")
    print(final_response.content)

else:

    print("\nThe model did not call the tool.")
    print("Model answer:")
    print(response.content)