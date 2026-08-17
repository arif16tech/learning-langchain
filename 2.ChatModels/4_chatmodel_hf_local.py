# Open source model on local machine

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen3-1.7B",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 100,
        "temperature": 0.5
        }
)

model = ChatHuggingFace(llm=llm)

res = model.invoke("what is the capital of china?")

print(res.content)