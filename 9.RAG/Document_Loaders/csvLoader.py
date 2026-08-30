import csv
from langchain_core.documents import Document
import os

# path where the file is located
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "students.csv")

documents = []

# reading the file
with open(file_path, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        documents.append(
            Document(
                page_content=str(row),
                metadata={"source": "students.csv"}
            )
        )

print("Number of documents:", len(documents))

print(documents[0].page_content)
print(documents[0].metadata)