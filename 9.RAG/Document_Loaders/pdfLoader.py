import pypdf
from langchain_core.documents import Document
import os

# path of the file
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "sample1.pdf")

# reading the file
reader = pypdf.PdfReader(file_path)

documents = []

for page in reader.pages:
    text = page.extract_text()

    documents.append(
        Document(
            page_content=text,
            metadata={"source": "sample1.pdf"}
        )
    )

# print number of pages
print("Number of pages:", len(documents))

# print first page content and metadata
print(documents[0].page_content)
print(documents[0].metadata)