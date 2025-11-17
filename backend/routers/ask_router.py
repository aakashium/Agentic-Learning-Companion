from fastapi import APIRouter, Query
from backend.vectorstore.chroma_client import ChromaVectorStore
from backend.models.llm_client import LLMClient
from backend.services.rag_engine import RAGEngine

router = APIRouter()

vectorstore = ChromaVectorStore()
llm = LLMClient()
rag = RAGEngine()  

@router.get("/")
def ask_question(q: str = Query(...), k: int = 5):
    """
    Hybrid RAG answer:
    - BM25 lexical ranking
    - SentenceTransformer semantic ranking
    - Combined scores
    """
    result = rag.answer(q, k=k)

    return {
        "query": q,
        "answer": result["answer"],
        "retrieved": result["retrieved"]
    }