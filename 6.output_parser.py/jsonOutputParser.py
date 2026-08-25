# In this i learn about json output parser from langchain 
# its help to generate the structured output from the model 

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3.8-2.4T-A95B",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template='Give me the name, age and city of a fictional person \n {formate_instruction}',
    input_variables=[],
    partial_variables={
        'formate_instruction': parser.get_format_instructions()
    }
)

chain = template | model | parser

result = chain.invoke({})

print(result)
print(type(result)) 