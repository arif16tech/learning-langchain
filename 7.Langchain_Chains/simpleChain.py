# In this i learned how to create a chain using pipe operator | 

from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template="Tell me 5 jokes about {topic}", 
    input_variables=["topic"]
)

model = ChatGroq(model="openai/gpt-oss-120b")

parser = StrOutputParser()

chain = prompt | model | parser

response = chain.invoke({"topic": "human"})

print(response)

chain.get_graph().print_ascii()