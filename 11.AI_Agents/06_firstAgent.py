from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

# 1. Create Tavily tool
search_tool = TavilySearch(max_results=2)

# 2. Create LLM
model = ChatGroq(model="openai/gpt-oss-120b")

# 3. Create Agent
agent = create_agent(
    model=model,
    tools=[search_tool],
)

# 4. Run Agent
response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Search the web for the latest tech news."
        }
    ]
})


# 5. Print final response
print("\n-----Response-----")
print(response["messages"])

print("\n-----Final Response-----")
# print(response["messages"][-1].content)