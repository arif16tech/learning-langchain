# web search using tavily tool

from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()

search = TavilySearch(max_results=1)

result = search.invoke({
    "query": "today news in india"
})

print(result)