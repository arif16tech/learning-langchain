# In this i learn about conditional chain in langchain, it will run two chains in conditional and then return the response based on the condition result.

from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

# define the models
model = ChatGroq(model="openai/gpt-oss-120b")

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="The sentiment of the feedback.")

parser = StrOutputParser()
parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate.from_template("Classify the sentiment of the following text into positive or negative:\n {feedback} \n {format_instructions}")

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate.from_template("Write an appropriate response to this positive feedback \n {feedback}")
prompt3 = PromptTemplate.from_template("Write an appropriate response to this negative feedback \n {feedback}")

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | model | parser),
    (lambda x: x.sentiment == "negative", prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain

result = chain.invoke({'feedback': 'This is a worst phone', "format_instructions": parser2.get_format_instructions()})

print(result)

chain.get_graph().print_ascii()