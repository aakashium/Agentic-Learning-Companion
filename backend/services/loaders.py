from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_core.documents import Document
import tempfile

def load_text(text: str):
    return [Document(page_content=text)]

def load_url(url: str):
    loader = WebBaseLoader(web_paths=[url])
    return loader.load()

def load_pdf(pdf_bytes: bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp.flush()
        loader = PyPDFLoader(tmp.name)
        return loader.load()
