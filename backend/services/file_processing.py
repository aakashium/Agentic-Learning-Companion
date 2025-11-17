from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_core.documents import Document
import tempfile
from backend.config import settings


# ---------------------------------------------------------
# 1) Load plain text
# ---------------------------------------------------------
def load_plain_text(text: str):
    """
    Convert raw text into a single LangChain Document.

    Parameters:
        text (str): User-provided text.

    Returns:
        List[Document]: A list with one Document.
    """
    return [
        Document(
            page_content=text,
            metadata={"source": "user_text"}
        )
    ]


# ---------------------------------------------------------
# 2) Load PDF files (bytes)
# ---------------------------------------------------------
def load_pdf(pdf_bytes: bytes):
    """
    Load a PDF into LangChain Documents.

    We temporarily save the bytes into a file because
    PyPDFLoader requires a file path.

    Parameters:
        pdf_bytes (bytes): Raw uploaded PDF content.

    Returns:
        List[Document]: Parsed pages as Documents.
    """
    try:
        # Save PDF to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()
            temp_path = tmp.name

        # Load PDF using LangChain
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        # Add metadata
        for d in docs:
            d.metadata["source"] = "pdf_file"

        return docs

    except Exception as e:
        # If anything fails, return a text document describing the error
        return [
            Document(page_content=f"PDF error: {e}", metadata={"error": True})
        ]


# ---------------------------------------------------------
# 3) Load a webpage
# ---------------------------------------------------------
def load_url(url: str, user_agent: str = "Agentic-Learning-Companion/1.0"):
    """
    Fetch and load webpage content using WebBaseLoader.

    Parameters:
        url (str): The webpage URL.
        user_agent (str): Custom user agent header to avoid warnings.

    Returns:
        List[Document]: Extracted text from the webpage.
    """
    try:
        loader = WebBaseLoader(
            web_paths=[url],
            header_template={"User-Agent": user_agent}
        )

        docs = loader.load()

        # Tag each chunk with source URL
        for d in docs:
            d.metadata["source"] = url

        return docs

    except Exception as e:
        return [
            Document(
                page_content=f"URL error: {e}",
                metadata={"source": url, "error": True}
            )
        ]
