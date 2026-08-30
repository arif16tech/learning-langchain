from langchain_core.documents import Document
import os

# path where the file is located
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "sample.txt")

# reading the file
with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

# creating a Document object
document = Document(
    page_content=text,
    metadata= {"source": "sample.txt"}
)

print(document)
print(type(document))