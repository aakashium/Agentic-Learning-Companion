from fastapi import APIRouter, UploadFile, File, Form
from backend.services.loaders import load_text, load_url, load_pdf
from backend.services.chunker import chunk_docs
from backend.vectorstore import VectorStore

router = APIRouter()
vs = VectorStore()

@router.post("/text")
def ingest_text(text: str = Form(...)):
    docs = chunk_docs(load_text(text))
    vs.add(docs)
    return {"status": "ok", "chunks": len(docs)}

@router.post("/url")
def ingest_url(url: str = Form(...)):
    docs = chunk_docs(load_url(url))
    vs.add(docs)
    return {"status": "ok", "chunks": len(docs)}

@router.post("/pdf")
def ingest_pdf(file: UploadFile = File(...)):
    docs = chunk_docs(load_pdf(file.file.read()))
    vs.add(docs)
    return {"status": "ok", "chunks": len(docs)}
