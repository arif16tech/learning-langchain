import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

url = "https://example.com"

response = requests.get(url)

soup = BeautifulSoup(response.text, 'html.parser')

text = soup.get_text(separator="\n", strip=True)

document = Document(
    page_content=text,
    metadata={"source": url}
)

print(document.page_content)
print(document.metadata)