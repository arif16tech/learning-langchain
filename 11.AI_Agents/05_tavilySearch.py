from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain.messages import HumanMessage, ToolMessage


load_dotenv()


# 1. Create Tavily Search Tool

search_tool = TavilySearch(
    max_results=3
)

# 2. Create LLM

model = ChatGroq(
    model="openai/gpt-oss-120b"
)

# 3. Give Search Tool to LLM

model_with_tools = model.bind_tools(
    [search_tool]
)

# 4. User Message

user_message = HumanMessage(
    content="Use the web search tool to find the latest AI news."
)

# 5. LLM

response = model_with_tools.invoke([
    user_message
])


print("\n--- TOOL CALL ---")
print(response.tool_calls)

# 6. Check if Tool Was Called

if response.tool_calls:

    tool_call = response.tool_calls[0]

    # 7. Execute Tavily

    tool_result = search_tool.invoke(
        tool_call["args"]
    )


    print("\n--- TAVILY RESULT ---")
    print(tool_result)

    # 8. Create ToolMessage

    tool_message = ToolMessage(
        content=str(tool_result),
        tool_call_id=tool_call["id"]
    )

    # 9. Send Result Back to LLM

    final_response = model_with_tools.invoke([
        user_message,
        response,
        tool_message
    ])

    # 10. Final Answer

    print("\n--- FINAL ANSWER ---")
    print(final_response.content)


else:

    print("\nThe model did not call the search tool.")
    print(response.content)