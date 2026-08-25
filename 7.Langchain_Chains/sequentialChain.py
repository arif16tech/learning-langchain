# In this i learn about sequential chain in langchain, it will run two chains in sequential and then combine the results.

from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

load_dotenv()

# define the model
model = ChatGroq(model="openai/gpt-oss-120b")

# define the prompt
prompt1 = PromptTemplate.from_template(
    "Explain {topic} in very simple words."
)

prompt2 = PromptTemplate.from_template(
    "Based on this explanation: \n {explanation} Generate 3 simple questions."
)

# define the parser
parser = StrOutputParser()

# define the chain
# RunnableLambda is used to map the output of the first chain to the input of the second chain
chain1 = prompt1 | model | parser
chain2 = prompt2 | model | parser

chain = chain1 | RunnableLambda(lambda x: {"explanation": x}) | chain2

response = chain.invoke({"topic": "Unemployment in india"})

print(response)