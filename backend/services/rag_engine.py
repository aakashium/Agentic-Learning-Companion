import numpy as np

from backend.vectorstore.chroma_client import ChromaVectorStore
from backend.services.bm25_search import bm25_search
from backend.models.llm_client import LLMClient


class RAGEngine:
    def __init__(self, w_vec: float = 0.6, w_bm25: float = 0.4):
        self.vectorstore = ChromaVectorStore()
        self.llm = LLMClient()
        self.w_vec = w_vec
        self.w_bm25 = w_bm25

    def _normalize(self, scores):
        arr = np.array(scores, dtype=float)
        if arr.size == 0:
            return arr
        mn, mx = arr.min(), arr.max()
        if abs(mx - mn) < 1e-9:
            return np.ones_like(arr)
        return (arr - mn) / (mx - mn + 1e-9)

    def hybrid_retrieve(self, query: str, k: int = 5):
        # Vector search
        vec_results = self.vectorstore.retrieve_with_scores(query, k)
        vec_docs = []
        for doc, score in vec_results:
            vec_docs.append({
                "doc_id": doc.metadata.get("doc_id"),
                "content": doc.page_content,
                "vec_score": float(score)
            })

        # BM25 search
        bm_results = bm25_search(query, top_k=k)

        # Normalize scores
        vec_norm = self._normalize([v["vec_score"] for v in vec_docs])
        bm_norm = self._normalize([b["score"] for b in bm_results])

        # Merge scores
        merged = {}
        # Vector
        for i, d in enumerate(vec_docs):
            doc_id = d["doc_id"]
            merged[doc_id] = {
                "content": d["content"],
                "vec_score": float(vec_norm[i]),
                "bm25_score": 0.0
            }

        # BM25
        for i, d in enumerate(bm_results):
            doc_id = d["doc_id"]
            if doc_id not in merged:
                merged[doc_id] = {
                    "content": d["content"],
                    "vec_score": 0.0,
                    "bm25_score": float(bm_norm[i])
                }
            else:
                merged[doc_id]["bm25_score"] = float(bm_norm[i])

        # Compute combined scores
        final = []
        for doc_id, vals in merged.items():
            combined = self.w_vec * vals["vec_score"] + self.w_bm25 * vals["bm25_score"]
            final.append({
                "doc_id": doc_id,
                "content": vals["content"],
                "vec_score": vals["vec_score"],
                "bm25_score": vals["bm25_score"],
                "combined_score": combined
            })

        # Sort by combined score
        final_sorted = sorted(final, key=lambda x: x["combined_score"], reverse=True)[:k]
        return final_sorted

    def answer(self, question: str, k: int = 5):
        retrieved = self.hybrid_retrieve(question, k=k)

        context = "\n\n".join(
            [f"[{i+1}] {item['content']}" for i, item in enumerate(retrieved)]
        )

        prompt = f"""
Use ONLY the context below to answer the question.
Cite using [1], [2], [3].

Context:
{context}

Question: {question}

Answer:
"""
        answer = self.llm.generate(prompt)

        return {
            "answer": answer,
            "retrieved": retrieved
        }
