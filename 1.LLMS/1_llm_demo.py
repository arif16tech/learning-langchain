# in this i have used chat model because i don't have openAI api key.

from langchain_openai import OpenAI # this is LLMs not Chat Models
from langchain_groq import ChatGroq # this is Chat Models
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b")

response = llm.invoke("what is the capital of india")

print(response.content)