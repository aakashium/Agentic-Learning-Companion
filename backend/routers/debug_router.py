from fastapi import APIRouter, Query
from backend.vectorstore.chroma_client import ChromaVectorStore
from backend.services.bm25_search import bm25_search
from backend.services.rag_engine import RAGEngine

router = APIRouter()

vectorstore = ChromaVectorStore()
rag = RAGEngine()


@router.get("/retrieval")
def debug_retrieval(q: str = Query(...), k: int = 5):
    # Vector-only results
    vec_raw = vectorstore.retrieve_with_scores(q, k)
    vector_results = [
        {
            "doc_id": doc.metadata.get("doc_id"),
            "score": float(score),
            "content": doc.page_content
        }
        for doc, score in vec_raw
    ]

    # BM25-only results
    bm25_results = bm25_search(q, top_k=k)

    # Hybrid results
    hybrid_results = rag.hybrid_retrieve(q, k)

    return {
        "query": q,
        "vector_results": vector_results,
        "bm25_results": bm25_results,
        "hybrid_results": hybrid_results
    }
