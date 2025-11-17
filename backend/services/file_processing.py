from langchain_community.document_loaders import (
    PyPDFLoader, 
    WebBaseLoader
)
from langchain_core.documents import Document


def load_plain_text(text: str):
    """Load plaintext as a LangChain Document."""
    return [Document(page_content=text, metadata={"source": "user_text"})]


def load_pdf(pdf_bytes: bytes):
    """Load PDF using LangChain PyPDFLoader."""
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()
            loader = PyPDFLoader(tmp.name)
            docs = loader.load()
            for d in docs:
                d.metadata["source"] = "pdf_file"
            return docs
    except Exception as e:
        return [Document(page_content=f"PDF error: {e}", metadata={})]


def load_url(url: str):
    """Load webpage using LangChain WebBaseLoader."""
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = url
        return docs
    except Exception as e:
        return [Document(page_content=f"URL error: {e}", metadata={})]
