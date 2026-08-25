# In this i learn about parallel chain in langchain, it will run two chains in parallel and then combine the results.

from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen3-1.7B",
    task="text-generation"
)

# define the models
model1 = ChatGroq(model="openai/gpt-oss-120b")
model2 = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate.from_template("Explain {topic} from a technical perspective.\nFocus on important technical concepts.")

prompt2 = PromptTemplate.from_template("Explain {topic} to a complete beginner.\nUse very simple language and examples.")

prompt3 = PromptTemplate.from_template("""You are given two explanations of the same topic.

Technical explanation:
{technical}

Beginner explanation:
{beginner}

Combine both explanations into one clear and complete answer.
Keep the technical accuracy but make it easy to understand.""")

parser = StrOutputParser()

# parallel chain
chain1 = prompt1 | model1 | parser
chain2 = prompt2 | model2 | parser

parallel_chain = RunnableParallel(
    technical= chain1,
    beginner= chain2
)

final_chain = parallel_chain | prompt3 | model2 | parser

response = final_chain.invoke({"topic": "How does a REST API work?"})

print(response)

final_chain.get_graph().print_ascii()