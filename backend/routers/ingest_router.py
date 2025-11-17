from fastapi import APIRouter, UploadFile, File, Form
from uuid import uuid4
import os

from backend.services.file_processing import load_url, load_pdf, load_plain_text
from backend.services.chunker import chunk_documents
from backend.vectorstore.chroma_client import ChromaVectorStore
from backend.services.bm25_search import index_documents
from backend.config import settings

router = APIRouter()
vectorstore = ChromaVectorStore()


# URL INGESTION 
@router.post("/url")
async def ingest_url(url: str = Form(...)):
    try:
        docs = load_url(url, user_agent=settings.USER_AGENT)
        chunks = chunk_documents(docs)

        bm25_docs = []
        for chunk in chunks:
            doc_id = chunk.metadata.get("doc_id", f"doc_{uuid4().hex}")
            chunk.metadata["doc_id"] = doc_id

            bm25_docs.append({
                "doc_id": doc_id,
                "content": chunk.page_content,
            })

        vectorstore.add_documents(chunks)
        index_documents(bm25_docs)

        return {"message": "URL ingested", "chunks_added": len(chunks)}

    except Exception as e:
        return {"error": str(e), "url": url}


# =============== PDF INGESTION ===================
@router.post("/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    try:
        temp_path = f"temp_{uuid4()}.pdf"

        with open(temp_path, "wb") as f:
            f.write(await file.read())

        docs = load_pdf(temp_path)
        chunks = chunk_documents(docs)

        bm25_docs = []
        for chunk in chunks:
            doc_id = chunk.metadata.get("doc_id", f"doc_{uuid4().hex}")
            chunk.metadata["doc_id"] = doc_id

            bm25_docs.append({
                "doc_id": doc_id,
                "content": chunk.page_content,
            })

        vectorstore.add_documents(chunks)
        index_documents(bm25_docs)

        os.remove(temp_path)

        return {"message": "PDF ingested", "chunks_added": len(chunks)}

    except Exception as e:
        return {"error": str(e)}


# =============== TEXT INGESTION ===================
@router.post("/text")
async def ingest_text(text: str = Form(...)):
    try:
        docs = load_text(text)
        chunks = chunk_documents(docs)

        bm25_docs = []
        for chunk in chunks:
            doc_id = chunk.metadata.get("doc_id", f"doc_{uuid4().hex}")
            chunk.metadata["doc_id"] = doc_id

            bm25_docs.append({
                "doc_id": doc_id,
                "content": chunk.page_content,
            })

        vectorstore.add_documents(chunks)
        index_documents(bm25_docs)

        return {"message": "Text ingested", "chunks_added": len(chunks)}

    except Exception as e:
        return {"error": str(e)}
