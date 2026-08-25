# Sementic search in documents using embeddings models locally 

from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Viral kohli is an Indian Cricketer known for his aggressive style of play and chasing down totals.",
    "The Taj Mahal is an ivory white marble mausoleum on the south bank of the Yamuna river in the Indian city of Agra.",
    "Lionel Messi is an Argentine professional footballer",
    "Sachin Tendulkar is a former Indian international cricketer and was the captain of the Indian national cricket team.",
    "Neymar is a Brazilian professional footballer",
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
    "Cristiano Ronaldo is a Portuguese professional footballer"
]

query = "tell me about messi"

doc_embedding = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embedding)[0]

index, score = sorted(list(enumerate(scores)), key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print("Similarity Score: ", score)
