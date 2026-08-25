# In this i learn about pydantic output parser from langchain 
# its help to generate the structured output from the model 

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

# define the model
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3.8-2.4T-A95B",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):

    name: str = Field(description="the name of the person")
    age: int = Field(gt=18, description="the age of the person")
    city: str = Field(description="the city of the person")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate the name, age and city of a fictinal {place} person \n {formate_instruction}',
    input_variables=['place'],
    partial_variables={
        'formate_instruction': parser.get_format_instructions()
    }
)

chain = template | model | parser

result = chain.invoke({"place": "Kashmir"})

print(result)