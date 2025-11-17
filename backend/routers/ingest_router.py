from fastapi import APIRouter, UploadFile, File, Form
from uuid import uuid4

from backend.services.file_processing import load_pdf, load_url, load_plain_text
from backend.services.chunker import chunk_documents
from backend.vectorstore.chroma_client import ChromaVectorStore
from backend.services.bm25_search import index_documents

router = APIRouter()
vectorstore = ChromaVectorStore()


@router.post("/text")
async def ingest_text(text: str = Form(...)):
    docs = load_plain_text(text)
    chunks = chunk_documents(docs)

    # assign doc_ids + prepare BM25 docs
    bm25_docs = []
    for c in chunks:
        if "doc_id" not in c.metadata:
            c.metadata["doc_id"] = f"doc_{uuid4().hex}"
        bm25_docs.append({
            "doc_id": c.metadata["doc_id"],
            "content": c.page_content
        })

    vectorstore.add_documents(chunks)
    index_documents(bm25_docs)

    return {"message": "Text ingested", "chunks_added": len(chunks)}


@router.post("/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    docs = load_pdf(pdf_bytes)
    chunks = chunk_documents(docs)

    bm25_docs = []
    for c in chunks:
        if "doc_id" not in c.metadata:
            c.metadata["doc_id"] = f"doc_{uuid4().hex}"
        bm25_docs.append({
            "doc_id": c.metadata["doc_id"],
            "content": c.page_content
        })

    vectorstore.add_documents(chunks)
    index_documents(bm25_docs)

    return {"message": "PDF ingested", "chunks_added": len(chunks)}


@router.post("/url")
async def ingest_url(url: str = Form(...)):
    docs = load_url(url)
    chunks = chunk_documents(docs)

    bm25_docs = []
    for c in chunks:
        if "doc_id" not in c.metadata:
            c.metadata["doc_id"] = f"doc_{uuid4().hex}"
        bm25_docs.append({
            "doc_id": c.metadata["doc_id"],
            "content": c.page_content
        })

    vectorstore.add_documents(chunks)
    index_documents(bm25_docs)

    return {"message": "URL ingested", "chunks_added": len(chunks)}
