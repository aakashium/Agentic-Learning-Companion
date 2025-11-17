from backend.services.bm25_search import bm25_search
from backend.vectorstore.chroma_client import ChromaVectorStore
from backend.services.rag_engine import RAGEngine
from backend.eval.retrieval_eval import evaluate_retrieval

vectorstore = ChromaVectorStore()
rag = RAGEngine()

def evaluate_all_retrievers(query, gold, k=5):
    # Vector only
    vec_results = vectorstore.retrieve_with_scores(query, k)
    vec_docs = [{"doc_id": doc.metadata.get("doc_id"), "content": doc.page_content}
                 for doc, score in vec_results]

    # BM25 only
    bm25_results = bm25_search(query, top_k=k)

    # Hybrid
    hybrid_results = rag.hybrid_retrieve(query, k)

    return {
        "vector": evaluate_retrieval(gold, vec_docs, k),
        "bm25": evaluate_retrieval(gold, bm25_results, k),
        "hybrid": evaluate_retrieval(gold, hybrid_results, k),
    }
