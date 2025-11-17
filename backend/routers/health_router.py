from fastapi import APIRouter
import os

from backend.vectorstore.chroma_client import ChromaVectorStore
from backend.services.bm25_search import _get_or_create_index

router = APIRouter()

@router.get("/")
def health_check():
    # Check Chroma DB
    try:
        vs = ChromaVectorStore()
        chroma_ok = True
    except Exception:
        chroma_ok = False

    # Check BM25 Index
    try:
        ix = _get_or_create_index()
        bm25_ok = ix is not None
    except Exception:
        bm25_ok = False

    # Embedding model
    try:
        embed_model = vs.embedder
        embed_ok = True
    except Exception:
        embed_ok = False

    return {
        "status": "ok" if (chroma_ok and bm25_ok and embed_ok) else "error",
        "chroma": chroma_ok,
        "bm25": bm25_ok,
        "embeddings": embed_ok
    }
